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

#include "GvvTransferFunctionInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Tag name identifying a space profile element
 */
const char* GvvTransferFunctionInterface::cTypeName = "TransferFunction";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvTransferFunctionInterface::GvvTransferFunctionInterface()
:	GvvBrowsable()
,	_filename()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvTransferFunctionInterface::~GvvTransferFunctionInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getTypeName() const
{
	//return "PIPELINE";
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getName() const
{
	return "TransferFunction";
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool GvvTransferFunctionInterface::initialize()
{
	return false;
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool GvvTransferFunctionInterface::finalize()
{
	return false;
}

/******************************************************************************
 * Get the associated filename
 *
 * @return the associated filename
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getFilename() const
{
	return _filename.c_str();
}

/******************************************************************************
 * Set the associated filename
 *
 * @param pFilename the associated filename
 ******************************************************************************/
void GvvTransferFunctionInterface::setFilename( const char* pFilename )
{
	_filename = std::string( pFilename );
}


/******************************************************************************
 * Update the associated transfer function
 *
 * @param the new transfer function data
 * @param the size of the transfer function
 ******************************************************************************/
void GvvTransferFunctionInterface::update( float* pData, unsigned int pSize )
{
}
