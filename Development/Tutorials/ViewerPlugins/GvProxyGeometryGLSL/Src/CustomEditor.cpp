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

#include "CustomEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "CustomSectionEditor.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Creator function
 *
 * @param pParent parent widget
 * @param pBrowsable pipeline element from which the editor will be associated
 *
 * @return the editor associated to the GigaVoxels pipeline
 ******************************************************************************/
GvvEditor* CustomEditor::create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable )
{
	return new CustomEditor( pParent );
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
CustomEditor::CustomEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvPipelineEditor( pParent, pFlags )
,	_customSectionEditor( NULL )
{
	// Create the user custom editor
	_customSectionEditor = new CustomSectionEditor( pParent, pFlags );
	assert( _customSectionEditor != NULL );
	if ( _customSectionEditor != NULL )
	{
		// Store the user custom editor
		_sectionEditors.push_back( _customSectionEditor );
	}
	else
	{
		// TO DO handle error
		// ...
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomEditor::~CustomEditor()
{
}
