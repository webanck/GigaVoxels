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

#ifndef GVVEDITORWINDOW_H
#define GVVEDITORWINDOW_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQEditorWidget.h"
#include "GvvContextListener.h"

// Qt
#include <QWidget>
#include <QHash>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvBrowsable;
}
namespace GvViewerGui
{
	class GvvEditor;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{
	/**
	 * Factory method used to create editors.
	 * Each editor need to provide a static method exactly as this function pointer.
	 */
	typedef GvvEditor* (GvvEditorFactory)( QWidget*, GvViewerCore::GvvBrowsable* );
}

namespace GvViewerGui
{

/**
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvEditorWindow : public QWidget, public Ui::GvvQEditorWidget, public GvvContextListener
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 */
	GvvEditorWindow( QWidget* pParent = NULL );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvEditorWindow();

	/**
	 * Edits the specified editable entity
	 *
	 * @param	pEditable	the entity to be edited
	 */
	void edit( GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Clears this editor
	 */
	void clear();

	/**
	 * Registers the specified editor builder
	 *
	 * @param pBuilder the editor builder to be registered
	 */
	void registerEditorFactory( const QString& pEditableType, GvvEditorFactory* pEditorFactory );

	/**
	 * Unregisters the specified editor builder
	 *
	 * @param pBuilder the editor builder to be unregistered
	 */
	void unregisterEditorFactory( const QString& pEditableType );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/
	
	/**
	 * Type definition of factory methods used to create editors
	 */
	typedef QHash< unsigned int, GvvEditorFactory* > GvvEditorFactories;
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The factory methods used to create editors
	 */
	GvvEditorFactories _editorFactories;

	/******************************** METHODS *********************************/

	/**
	 * This slot is called when the current browsable is changed
	 */
	virtual void onCurrentBrowsableChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/	

	/**
	 * The current editor
	 */ 
	GvvEditor* _currentEditor;

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvEditorWindow( const GvvEditorWindow& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvEditorWindow& operator=( const GvvEditorWindow& );
	
};

} // namespace GvViewerGui

#endif
