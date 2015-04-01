/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
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

#ifndef _RLE_COMPRESSOR_H_
#define _RLE_COMPRESSOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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
 * @class GvCommonGraphicsPass
 *
 * @brief The GvCommonGraphicsPass class provides interface to
 *
 * Some resources from OpenGL may be mapped into the address space of CUDA,
 * either to enable CUDA to read data written by OpenGL, or to enable CUDA
 * to write data for consumption by OpenGL.
 */
class RLECompressor
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	RLECompressor();

	/**
	 * Destructor
	 */
	virtual ~RLECompressor();

	/**
	 * Initiliaze
	 */
	void initialize();

	/**
	 * Finalize
	 */
	void finalize();

	/**
	 * ...
	 *
	 * @param pInput ...
	 * @param pOutput ...
	 */
	static void sameArray( const unsigned char* pInput, const unsigned char* pOutput );

	/**
	 * Given an input data array, write output array with RLE encoding (compression)
	 *
	 * @param pInput input data
	 * @param pOutput output data (RLE encoding - compression)
	 */
	//static void RLEcomp( const unsigned int* pInput, unsigned int* pOutput );
	//static void RLEcompBis( const unsigned char* pInput, unsigned char* pOutput );

	/**
	 * Given an input data array, write output arrays with RLE encoding (compression)
	 *
	 * @param input initial array
	 * @param nBricks the number of bricks in the input array
	 * @param bricksEnds array containing the ends of each bricks in the compressed arrays 
	 * @param plateausValues values of the plateaus in the bricks
	 * @param plateausStarts beginnings of each plateaus in the bricks
	 */
	void compressionPrefixSum( const unsigned int* input, 
			const unsigned int nBricks,
			unsigned int* bricksEnds, 
			unsigned int* plateausValues,
			unsigned char* plateausStarts );

	/**
	 * Given an input data array, write output array with RLE encoding (compression)
	 * - data
	 * - and offsets
	 *
	 * @param pInput input data
	 * @param pOutputData output data (RLE encoding - compression)
	 * @param pOutputOffset output offsets (RLE encoding - compression)
	 */
	static bool RLEcompOffset( const unsigned char* pInput, unsigned char* pOutputData, unsigned int* pOutputOffset );

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

	/**
	 * Copy constructor forbidden.
	 */
	RLECompressor( const RLECompressor& );

	/**
	 * Copy operator forbidden.
	 */
	RLECompressor& operator=( const RLECompressor& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RLECompressor.inl"

#endif
