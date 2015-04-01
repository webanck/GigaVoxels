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

#include "GvvEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSectionEditor.h"

// Qt
#include <QWidget>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor.
 ******************************************************************************/
GvvEditor::GvvEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	_sectionEditors()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvEditor::~GvvEditor()
{
	for ( unsigned int i = 0; i < getNbSections(); i++ )
	{
		GvvSectionEditor* sectionEditor = _sectionEditors[ i ];
		delete sectionEditor;
		sectionEditor = NULL;
	}
	_sectionEditors.clear();
}

/******************************************************************************
 * Get the number of sections.
 *
 * @return the number of sections
 ******************************************************************************/
unsigned int GvvEditor::getNbSections() const
{
	return _sectionEditors.size();
}

/******************************************************************************
 * ...
 ******************************************************************************/
GvvSectionEditor* GvvEditor::getSectionEditor( unsigned int pIndex )
{
	assert( pIndex < getNbSections() );
	
	GvvSectionEditor* sectionEditor = NULL;

	if ( pIndex < getNbSections() )
	{
		sectionEditor = _sectionEditors[ pIndex ];
	}

	return sectionEditor;
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void GvvEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	for ( unsigned int i = 0; i < getNbSections(); i++ )
	{
		GvvSectionEditor* sectionEditor = _sectionEditors[ i ];
		if ( sectionEditor != NULL )
		{
			sectionEditor->populate( pBrowsable );
		}
	}
}
