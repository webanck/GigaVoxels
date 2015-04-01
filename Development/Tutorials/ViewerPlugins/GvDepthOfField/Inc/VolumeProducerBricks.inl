/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** 
 * @version 1.0
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

template<typename DataTList>
VolumeProducerBricks<DataTList>::VolumeProducerBricks(const std::string &name, const uint3 &datasize, const uint3 &blocksize, int bordersize, bool usecache)
{
	this->_numChannels=Loki::TL::Length<DataTList>::value;
	this->_volumeRes=datasize;
	this->_bricksRes=blocksize;
	this->_borderSize=bordersize;
	this->_mipMapOrder=2;
	this->_useCache=usecache;

	//Compute num mipmaps levels
	int dataresmin=mincc(this->_volumeRes.x, mincc(this->_volumeRes.y, this->_volumeRes.z));

	int blocksNumLevels		= (int)(log(float(this->_bricksRes.x))/log((float)(this->_mipMapOrder)));
	this->_numMipMapLevels	= (int)(log(float(dataresmin))/log((float)(this->_mipMapOrder)))+1 ;
	this->_numMipMapLevels	= this->_numMipMapLevels - blocksNumLevels;

	if (this->_numMipMapLevels < 1)
		this->_numMipMapLevels = 1;

	uint3 true_bricksRes=this->_bricksRes+make_uint3(2*this->_borderSize);

	this->makeFilesNames(name.c_str());

	if (this->_useCache)
	{
		for (int level = 0; level < _numMipMapLevels; level++)
		{
			////Indices////
			std::string fileNameIndex = _filesNames[(_numChannels + 1) * level];

			FILE *fileindex = fopen(fileNameIndex.c_str(), "rb");

			if(fileindex)
			{
#ifdef WIN32
				_fseeki64(fileindex, 0, SEEK_END);

				__int64 size = _ftelli64(fileindex);
				__int64 expectedSize = (__int64)powf(8.0f, (float)level) * sizeof(unsigned int);
#else
				fseeko(fileindex, 0, SEEK_END);

				off_t size = ftello(fileindex);
				off_t expectedSize = (off_t)powf(8.0f, (float)level) * sizeof(unsigned int);
#endif
				if (size != expectedSize)
				{
					std::cerr << "VolumeProducerBricks::VolumeProducerBricks: file size expected = " << expectedSize
						<< ", size returned = " << size << " for " << fileNameIndex << std::endl;
				}

				unsigned int *tmpcache = new unsigned int[size / 4];

#ifdef WIN32
				_fseeki64(fileindex, 0, SEEK_SET);
#else
				fseeko(fileindex, 0, SEEK_SET);
#endif

				if (fread(tmpcache, 1, (size_t)size, fileindex) != size)
				{
					std::cout << "VolumeProducerBricks::VolumeProducerBricks: Unable to open file " << this->_filesNames[level] << std::endl;
					this->_useCache = false;
				}

				fclose(fileindex);

				_blockIndexCache.push_back(tmpcache);
			}
			else
			{
				std::cout << "VolumeProducerBricks::VolumeProducerBricks : Unable to open file index " << fileNameIndex << std::endl;
			}

			for (size_t channel = 0; channel < _numChannels; channel++)
			{
				////Blocks/////
				FILE *file = fopen(this->_filesNames[(_numChannels + 1) * level + channel + 1].c_str(), "rb");

				if (file)
				{
#ifdef WIN32
					_fseeki64(file, 0, SEEK_END);
					__int64 size = _ftelli64(file);
#else
					fseeko(file, 0, SEEK_END);
					off_t size = ftello(file);
#endif
					unsigned char *tmpcache;

#if USE_GPUFETCHDATA
					cudaHostAlloc((void**)&tmpcache, size, cudaHostAllocMapped |  cudaHostAllocWriteCombined);
					
					void *deviceptr;
					cudaHostGetDevicePointer(&deviceptr, tmpcache, 0);
					std::cout<<"Device pointer host mem: "<<(uint)deviceptr<<"\n";
#else
					//cudaMallocHost((void**)&tmpcache, size); //pinned memory
					//cudaHostAlloc((void **)&tmpcache, size);
					tmpcache = new unsigned char[(size_t)size];
#endif
					GV_CHECK_CUDA_ERROR("VolumeProducerBricks::VolumeProducerBricks: cache alloc");

#ifdef WIN32
					_fseeki64(file, 0, SEEK_SET);
#else
					fseeko(file, 0, SEEK_SET);
#endif

					if (fread(tmpcache, 1, (size_t)size, file) != size)
					{
						std::cout << "VolumeProducerBricks::VolumeProducerBricks: Can't read file" << std::endl;
						this->_useCache = false;
					}
					fclose(file);

					_blockCache.push_back(tmpcache);
				}
				else
				{
					std::cout << "VolumeProducerBricks::VolumeProducerBricks: Unable to open file " << this->_filesNames[(_numChannels + 1) * level + channel + 1] << std::endl;
					this->_useCache = false;
				}
			}
		}
	}

}

template<typename DataTList>
VolumeProducerBricks<DataTList>::~VolumeProducerBricks()
{
	for(size_t level = 0; level < _blockCache.size(); level++)
	{
		if (_blockCache[level])
			cudaFree(_blockCache[level]);

		if (_blockIndexCache[level])
			delete [] _blockIndexCache[level];
	}
}

template<typename DataTList>
void VolumeProducerBricks<DataTList>::makeFilesNames(const char *filesname)
{
	std::string sourceFileName = std::string(filesname);
	std::string nodesFileNameExt = ".nodes";
	std::string bricksFileNameExt = ".bricks";

	for (int i = 0; i < _numMipMapLevels; i++)
	{
		std::stringstream ssNodes;

		ssNodes << sourceFileName << "_BR" << _bricksRes.x << "_B" << _borderSize << "_L" << i << nodesFileNameExt;
		_filesNames.push_back(ssNodes.str());

		FileNameBuilder fnb(_bricksRes.x, _borderSize, i, sourceFileName, bricksFileNameExt, _filesNames);
		GvCore::StaticLoop<FileNameBuilder, Loki::TL::Length<DataTList>::value - 1>::go(fnb);
	}
}

template<typename DataTList>
unsigned int VolumeProducerBricks<DataTList>::getBlockIndex(int level, const uint3 &bpos) const
{
	uint3 levelsize=getLevelRes(level);
	uint3 blocksinlevel=levelsize/this->_bricksRes;

	unsigned int indexval=0;

	if(_useCache)
	{
		uint indexpos=bpos.x + bpos.y*blocksinlevel.x + bpos.z*blocksinlevel.x*blocksinlevel.y;
		//----------------------------------------
		// TEST
		//std::cout << "indexpos = " << indexpos << ", bpos = " << bpos << std::endl;
		//----------------------------------------
		indexval=_blockIndexCache[level][indexpos];
	}
	else
	{
#ifdef WIN32
		__int64 indexpos=((__int64)bpos.x + (__int64)(bpos.y)*blocksinlevel.x + (__int64)(bpos.z)*blocksinlevel.x*blocksinlevel.y)*sizeof(unsigned int);
#else
		off_t indexpos=(off_t)bpos.x + (off_t)(bpos.y * blocksinlevel.x) + (off_t)(bpos.z*blocksinlevel.x*blocksinlevel.y)*sizeof(unsigned int);
#endif
		//index//
		std::string fileNameIndex=this->_filesNames[(_numChannels + 1) * level];
		FILE *fileindex=fopen(fileNameIndex.c_str(), "rb");
		if(fileindex)
		{
#ifdef WIN32
			_fseeki64(fileindex, indexpos, 0);
#else
			fseeko(fileindex, indexpos, 0);
#endif
			fread(&indexval, sizeof(unsigned int), 1, fileindex);
			fclose(fileindex);
		}
		else
		{
			std::cout<<"VolumeProducerBricks::getBlockIndex() : Unable to open file index "<<fileNameIndex<<"\n"; 
		}
	}

	return indexval;
}

template<typename DataTList>
bool VolumeProducerBricks<DataTList>::loadBrick(int level, const uint3 &bpos, GvCore::GPUPoolHost<GvCore::Array3D, DataTList> *data, size_t offsetInPool)
{
	uint3 levelsize=getLevelRes(level);
	uint3 blocksinlevel=levelsize/this->_bricksRes;

	uint3 trueBlocksRes=this->_bricksRes+make_uint3(2*this->_borderSize);
	size_t blockmemsize = (size_t)(trueBlocksRes.x*trueBlocksRes.y*trueBlocksRes.z );

	unsigned int indexval=getBlockIndex(level, bpos);

	if (indexval & 0x40000000U)
	{
		ChannelInitializer c(this, indexval, blockmemsize, level, data, offsetInPool);

		GvCore::StaticLoop<ChannelInitializer, Loki::TL::Length<DataTList>::value - 1>::go(c);

		return true;
	}
	else
	{
		return false;
	}
} 

template <typename DataTList>
typename VolumeProducer<DataTList>::VPRegionInfo VolumeProducerBricks<DataTList>::getRegion(const float3 &pos, const float3 &size, GvCore::GPUPoolHost<GvCore::Array3D, DataTList> *outVol, size_t offsetInPool)
{
	int level =	getDataLevel(size);
	uint3 levelRes = getLevelRes(level);

	uint3 coordsInLevel = getCoordsInLevel(level, pos);
	uint3 blockCoords = getBlockCoords(level, pos);
	uint3 blocksInLevel = levelRes / this->_bricksRes;

	if (blockCoords.x >= blocksInLevel.x || blockCoords.y >= blocksInLevel.y || blockCoords.z >= blocksInLevel.z)
	{
		std::cerr << "VolumeProducerBricks<T>::getRegion: Invalid position and/or size." << std::endl;
		std::cerr << "VolumeProducerBricks<T>::getRegion: posf = " << pos << ", sizef = " << size << std::endl;
		return VolumeProducerBricks<DataTList>::VP_CONST_REGION;
	}

	if (level >= 0 && level < _numMipMapLevels)
	{
		if (loadBrick(level, blockCoords, outVol, offsetInPool))
			return VolumeProducerBricks<DataTList>::VP_UNKNOWN_REGION;
		else
			return VolumeProducerBricks<DataTList>::VP_CONST_REGION; // Correct ?
	}
	else
	{
		std::cout << "VolumeProducerBricks<T>::getRegion: Invalid requested block dimensions" << std::endl;
		return VolumeProducerBricks<DataTList>::VP_CONST_REGION;
	}
}

template <typename DataTList>
typename VolumeProducer<DataTList>::VPRegionInfo VolumeProducerBricks<DataTList>::getRegionInfo(const float3 &posf, const float3 &sizef/*, T *constValueOut*/) {

	int level =	getDataLevel(sizef);
	uint3 bpos = getBlockCoords(level, posf);

	uint3 levelsize=getLevelRes(level);
	uint3 blocksinlevel=levelsize/this->_bricksRes;

	uint3 trueBlocksRes=this->_bricksRes+ make_uint3(2*this->_borderSize);
	//size_t blockmemsize = (size_t)(trueBlocksRes.x*trueBlocksRes.y*trueBlocksRes.z );

	if (level >= 0 && level < _numMipMapLevels)
	{
		unsigned int indexval=getBlockIndex(level, bpos);

		// if there is a brick
		if (indexval & 0x40000000U)
		{
			// if we are on a terminal node
			if (indexval & 0x80000000U)
				return VolumeProducerBricks<DataTList>::VP_UNKNOWN_REGION;
			else
				return VolumeProducerBricks<DataTList>::VP_NON_CONST_REGION;
		}
		else
			return VolumeProducerBricks<DataTList>::VP_CONST_REGION;
	}
	else
	{
		return VolumeProducerBricks<DataTList>::VP_CONST_REGION;
	}
}

template <typename DataTList>
uint VolumeProducerBricks<DataTList>::getRegionInfoNew(const float3 &posf, const float3 &sizef)
{
	int level =	getDataLevel(sizef);
	uint3 bpos = getBlockCoords(level, posf);
	uint3 lres = getLevelRes(level);
	uint3 blocksInLevel = lres / this->_bricksRes;

	if (bpos.x >= blocksInLevel.x || bpos.y >= blocksInLevel.y || bpos.z >= blocksInLevel.z)
	{
		std::cerr << "VolumeProducerBricks<T>::getRegionInfo: Invalid position and/or size." << std::endl;
		std::cerr << "VolumeProducerBricks<T>::getRegionInfo: posf = " << posf << ", sizef = " << sizef << std::endl;
		return 0;
	}

	if (level >= 0 && level < _numMipMapLevels)
		return (getBlockIndex(level, bpos) & 0xC0000000);

	return 0;
}
