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

namespace GvUtils
{

/******************************************************************************
 * Constructor
 *
 * @param brickSize Brick size
 * @param borderSize Border size
 * @param level Level
 * @param fileName File name
 * @param fileExt File extension
 * @param result List of built filenames
 ******************************************************************************/
template< typename TDataTypeList >
inline GvFileNameBuilder< TDataTypeList >
::GvFileNameBuilder( uint pBrickSize, uint pBorderSize, uint pLevel, const std::string& pFileName,
					  const std::string& pFileExt, std::vector< std::string >& pResult )
:	mBrickSize( pBrickSize )
,	mBorderSize( pBorderSize )
,	mLevel( pLevel )
,	mFileName( pFileName )
,	mFileExt( pFileExt )
,	mResult( &pResult )
{
}

/******************************************************************************
 * ...
 *
 * @param Loki::Int2Type< TChannel > ...
 ******************************************************************************/
template< typename TDataTypeList >
template< int TChannel >
inline void GvFileNameBuilder< TDataTypeList >
::run( Loki::Int2Type< TChannel > )
{
	// Typedef to access the channel in the data type list
	typedef typename Loki::TL::TypeAt< TDataTypeList, TChannel >::Result ChannelType;

	// Build filename according to GigaVoxels internal syntax
	std::stringstream filename;
	filename << mFileName << "_BR" << mBrickSize << "_B" << mBorderSize << "_L" << mLevel
		<< "_C" << TChannel << "_" << GvCore::typeToString< ChannelType >() << mFileExt;

	// Store generated filename
	mResult->push_back( filename.str() );
}

} // namespace GvUtils
