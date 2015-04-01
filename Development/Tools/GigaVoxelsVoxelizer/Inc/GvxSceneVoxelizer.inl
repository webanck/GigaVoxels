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

namespace Gvx
{

/******************************************************************************
 * Get the data file path
 *
 * @return the data file path
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFilePath() const
{
	return _filePath;
}

/******************************************************************************
 * Set the data file path
 *
 * @param pFilePath the data file path
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFilePath( const std::string& pFilePath )
{
	_filePath = pFilePath;
}

/******************************************************************************
 * Get the data file name
 *
 * @return the data file name
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFileName() const
{
	return _fileName;
}

/******************************************************************************
 * Set the data file name
 *
 * @param pFileName the data file name
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFileName( const std::string& pFileName )
{
	_fileName = pFileName;
}

/******************************************************************************
 * Get the data file extension
 *
 * @return the data file extension
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFileExtension() const
{
	return _fileExtension;
}

/******************************************************************************
 * Set the data file extension
 *
 * @param pFileExtension the data file extension
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFileExtension( const std::string& pFileExtension )
{
	_fileExtension = pFileExtension;
}

/******************************************************************************
 * Get the max level of resolution
 *
 * @return the max level of resolution
 ******************************************************************************/
inline unsigned int GvxSceneVoxelizer::getMaxResolution() const
{
	return _maxResolution;
}

/******************************************************************************
 * Set the max level of resolution
 *
 * @param pValue the max level of resolution
 ******************************************************************************/
inline void GvxSceneVoxelizer::setMaxResolution( unsigned int pValue )
{
	_maxResolution = pValue;
}

/******************************************************************************
 * Tell wheter or not normals generation is activated
 *
 * @return a flag telling wheter or not normals generation is activated
 ******************************************************************************/
inline bool GvxSceneVoxelizer::isGenerateNormalsOn() const
{
	return _isGenerateNormalsOn;
}

/******************************************************************************
 * Set the flag telling wheter or not normals generation is activated
 *
 * @param pFlag the flag telling wheter or not normals generation is activated
 ******************************************************************************/
inline void GvxSceneVoxelizer::setGenerateNormalsOn( bool pFlag )
{
	_isGenerateNormalsOn = pFlag;
}

/******************************************************************************
 * Get the brick width
 *
 * @return the brick width
 ******************************************************************************/
inline unsigned int GvxSceneVoxelizer::getBrickWidth() const
{
	return _brickWidth;
}

/******************************************************************************
 * Set the brick width
 *
 * @param pValue the brick width
 ******************************************************************************/
inline void GvxSceneVoxelizer::setBrickWidth( unsigned int pValue )
{
	_brickWidth = pValue;
}

/******************************************************************************
 * Get the data type of voxels
 *
 * @return the data type of voxels
 ******************************************************************************/
inline GvxDataTypeHandler::VoxelDataType GvxSceneVoxelizer::getDataType() const
{
	return _dataType;
}

/******************************************************************************
 * Set the data type of voxels
 *
 * @param pType the data type of voxels
 ******************************************************************************/
inline void GvxSceneVoxelizer::setDataType( GvxDataTypeHandler::VoxelDataType pType )
{
	_dataType = pType;
}

}
