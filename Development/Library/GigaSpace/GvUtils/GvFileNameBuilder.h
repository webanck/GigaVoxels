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

#ifndef GV_FILENAME_BUILDER_H
#define GV_FILENAME_BUILDER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <vector>

// System
#include <string>
#include <sstream>

// Loki
#include <loki/Typelist.h>

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

namespace GvUtils
{

/** 
 * @struct GvFileNameBuilder
 *
 * @brief The GvFileNameBuilder struct provides...
 *
 * ...
 */
template< typename TDataTypeList >
class GvFileNameBuilder
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param brickSize Brick size
	 * @param borderSize Border size
	 * @param level Level
	 * @param fileName File name
	 * @param fileExt File extension
	 * @param result List of built filenames
	 */
	inline GvFileNameBuilder( uint pBrickSize, uint pBorderSize, uint pLevel, const std::string& pFileName,
							const std::string& pFileExt, std::vector< std::string >& pResult );

	/**
	 * ...
	 *
	 * @param Loki::Int2Type< TChannel > ...
	 */
	template< int TChannel >
	inline void run( Loki::Int2Type< TChannel > );

	/**************************************************************************
	**************************** PROTECTED SECTION ***************************
	**************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	***************************** PRIVATE SECTION ****************************
	**************************************************************************/

private:

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
	 * Level of resolution
	 */
	uint mLevel;

	/**
	 * File name
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

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvFileNameBuilder.inl"

#endif // GV_FILENAME_BUILDER_H
