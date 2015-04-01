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

#ifndef _VOLUMEPRODUCER_BRICKS_H_
#define _VOLUMEPRODUCER_BRICKS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <vector>
#include <string>
#include <sstream>

// GigaVoxels
#include <GvCore/TypeHelpers.h>
#include <GvCore/Array3D.h>
#include <GvCore/vector_types_ext.h>

// project
#include "VolumeProducer.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class VolumeProducerBricks
 *
 * @brief The VolumeProducerBricks class provides...
 *
 * Interface of all Volume Producers.
 */
template< typename TDataTList >
class VolumeProducerBricks : public VolumeProducer< TDataTList >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * Constuctor
	 *
	 * @param name name
	 * @param size size
	 * @param blocksize block size
	 * @param cache cache
	 */
	VolumeProducerBricks(const std::string& name, const uint3& size, const uint3& blocksize, int _borderSize, bool cache = false );

	/**
	 * Destuctor
	 */
	~VolumeProducerBricks();

	/**
	 * ...
	 */
	typename VolumeProducer< TDataTList >::VPRegionInfo getRegion( const float3& pos, const float3& size, GvCore::GPUPoolHost< GvCore::Array3D, TDataTList >* outVol, size_t offsetInPool );
	/**
	 * ...
	 */
	typename VolumeProducer< TDataTList >::VPRegionInfo getRegionInfo( const float3& posf, const float3& sizef/*, TDataTList* constValueOut = NULL*/ );
	/**
	 * ...
	 */
	uint getRegionInfoNew( const float3& posf, const float3& sizef );

	/**
	 * ...
	 */
	uint3 getLevelRes( uint level ) const
	{
		// return _volumeRes /  (1 << level ); // WARNING: suppose mipMapOrder == 2 !
		return _bricksRes * ( 1 << level ); // WARNING: suppose mipMapOrder == 2 !
	}

	/**
	 * ...
	 */
	float3 getFeaturesSize() const
	{
		return make_float3( 1.0f ) / make_float3( _volumeRes );
	}

	/**
	 * ...
	 */
	template< typename ChannelType >
	void readBrick( int channel, unsigned int indexval, unsigned int blockmemsize, unsigned int level, GvCore::Array3D< ChannelType >* data, size_t offsetInPool )
	{
		unsigned int filepos = ( indexval & 0x3FFFFFFFU ) * blockmemsize * sizeof( ChannelType );

		if ( _useCache )
		{
			if ( _blockCache[ level * _numChannels + channel ] )
			{
				memcpy( data->getPointer( offsetInPool ), _blockCache[ level * _numChannels + channel ] + filepos, blockmemsize * sizeof( ChannelType ) );
			}
		}
		else
		{
			FILE* file = fopen( _filesNames[ level * ( _numChannels + 1 ) + channel + 1 ].c_str(), "rb" );

			if ( file )
			{
#ifdef WIN32
				_fseeki64( file, filepos, 0 );
#else
				fseeko( file, filepos, 0 );
#endif
				fread( data->getPointer( offsetInPool ), sizeof( ChannelType ), blockmemsize, file );
				fclose( file );
			}
		}
	}

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/** 
	 * @struct FileNameBuilder
	 *
	 * @brief The FileNameBuilder struct provides...
	 *
	 * ...
	 */
	struct FileNameBuilder
	{
		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/
		
		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/
				
		/**
		 * Constructor
		 *
		 * @param brickSize caller
		 * @param borderSize index value
		 * @param level block memory size
		 * @param fileName level
		 * @param fileExt data pool
		 * @param result offset in pool
		 */
		FileNameBuilder( uint brickSize, uint borderSize, uint level, const std::string& fileName,
						const std::string& fileExt, std::vector< std::string >& result )
		:	mBrickSize( brickSize )
		,	mBorderSize( borderSize )
		,	mLevel( level )
		,	mFileName( fileName )
		,	mFileExt( fileExt )
		,	mResult( &result )
		{
		}

		/**
		 * ...
		 *
		 * @param Loki::Int2Type< channel > index of the associated data pool's channel
		 */
		template< int channel >
		inline void run( Loki::Int2Type<channel> )
		{
			typedef typename Loki::TL::TypeAt< TDataTList, channel >::Result ChannelType;

			std::stringstream ss;

			ss << mFileName << "_BR" << mBrickSize << "_B" << mBorderSize << "_L" << mLevel
				<< "_C" << channel << "_" << GvCore::typeToString<ChannelType>() << mFileExt;

			mResult->push_back( ss.str() );
		}

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Brick size
		 */
		uint mBrickSize;

		/**
		 * Border size
		 */
		uint mBorderSize;

		/**
		 * Level
		 */
		uint mLevel;

		/**
		 * Filename
		 */
		std::string mFileName;

		/**
		 * File extension
		 */
		std::string mFileExt;

		/**
		 * List of built filenames
		 */
		std::vector< std::string >* mResult;

		/******************************** METHODS *********************************/

	};

	/** 
	 * @struct ChannelInitializer
	 *
	 * @brief The ChannelInitializer struct provides...
	 *
	 * ...
	 */
	struct ChannelInitializer
	{
		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/
		
		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Caller
		 */
		VolumeProducerBricks< TDataTList >* caller;

		/**
		 * Index value
		 */
		unsigned int indexval;

		/**
		 * Block memory size
		 */
		unsigned int blockmemsize;

		/**
		 * Level
		 */
		unsigned int level;

		/**
		 * Data pool
		 */
		GvCore::GPUPoolHost< GvCore::Array3D, TDataTList >* dataPool;

		/**
		 * Offset in pool
		 */
		size_t offsetInPool;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param pC caller
		 * @param pIval index value
		 * @param pBsize block memory size
		 * @param pL level
		 * @param pDp data pool
		 * @param pOip offset in pool
		 */
		ChannelInitializer( VolumeProducerBricks< TDataTList >* pC, unsigned int pIval, unsigned int pBsize, unsigned int pL, GvCore::GPUPoolHost< GvCore::Array3D, TDataTList >* pDp, size_t pOip )
		:	caller( pC )
		,	indexval( pIval )
		,	blockmemsize( pBsize )
		,	level( pL )
		,	dataPool( pDp )
		,	offsetInPool( pOip )
		{
		}

		/**
		 * ...
		 *
		 * @param Loki::Int2Type< channel > index of the associated data pool's channel
		 */
		template< int channel >
		inline void run( Loki::Int2Type< channel > )
		{
			typedef typename Loki::TL::TypeAt< TDataTList, channel >::Result ChannelType;

			GvCore::Array3D< ChannelType >* dataArray = dataPool->template getChannel< channel >();

			caller->readBrick< ChannelType >( channel, indexval, blockmemsize, level, dataArray, offsetInPool );
		}

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

	};

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Volume resolution
	 */
	uint3 _volumeRes;

	/**
	 * Brick resolution
	 */
	uint3 _bricksRes;
	/**
	 * Border size
	 */
	int _borderSize;

	/**
	 * Number of mipmap levels
	 */
	int _numMipMapLevels;
	/**
	 * Mipmap order
	 */
	int _mipMapOrder;

	/**
	 * Flag to tell wheter or not a cache is used
	 */
	bool _useCache;
	std::vector< unsigned char* > _blockCache;
	std::vector< unsigned int* > _blockIndexCache;

	/**
	 * Filenames
	 */
	std::vector< std::string > _filesNames;

	/**
	 * Number of channels
	 */
	size_t _numChannels;

	/******************************** METHODS *********************************/

	/**
	 * Return coords of the bloc in the blocks grid of the level.
	 */
	uint3 getBlockCoords( int level, const float3& posf ) const
	{
		return getCoordsInLevel( level, posf ) / this->_bricksRes;
	}

	/**
	 * ...
	 */
	int getDataLevel( const float3& sizef ) const
	{
		uint3 numNodes = make_uint3( 1.0f / sizef );
		int level = (int)( log( (float)numNodes.x ) / log( (float)( _mipMapOrder ) ) );
		return level;
	}

	/**
	 * ...
	 */
	uint3 getCoordsInLevel( int level, const float3& posf ) const
	{
		uint3 lres = getLevelRes( level );
		uint3 coordsinlevel = make_uint3( make_float3( lres ) * posf );
		return coordsinlevel;
	}

	/**
	 * ...
	 */
	void makeFilesNames( const char* filesname );

	/**
	 * ...
	 */
	unsigned int getBlockIndex( int level, const uint3& bpos ) const;

	/**
	 * ...
	 */
	bool loadBrick( int level, const uint3& bpos, GvCore::GPUPoolHost< GvCore::Array3D, TDataTList >* data, size_t offsetInPool );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
**************************************************************************/

#include "VolumeProducerBricks.inl"

#endif
