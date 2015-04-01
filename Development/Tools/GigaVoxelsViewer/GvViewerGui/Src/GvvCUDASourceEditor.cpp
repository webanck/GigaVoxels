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

#include "GvvCUDASourceEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Viewer
#include "GvvCUDASyntaxHighlighter.h"

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
 * Default constructor
 ******************************************************************************/
GvvCUDASourceEditor::GvvCUDASourceEditor( QWidget* pWidget )
:	QWidget( 0, 0 )
{
	_ui.setupUi( this );

	GvvCUDASyntaxHighlighter* lHighLight = new GvvCUDASyntaxHighlighter( _ui._textEdit->document() );

	connect( _ui._applyButton, SIGNAL( clicked() ), this, SLOT( onApply() ) );
	connect( _ui._compileButton, SIGNAL( clicked() ), this, SLOT( onCompile() ) );

	setAcceptDrops( true );
}

/******************************************************************************
 * Apply action
 ******************************************************************************/
void GvvCUDASourceEditor::onApply()
{
}

/******************************************************************************
 * Compile action
 ******************************************************************************/
void GvvCUDASourceEditor::onCompile()
{
}
