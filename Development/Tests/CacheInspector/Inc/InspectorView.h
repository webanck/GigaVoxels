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

#ifndef _INSPECTOR_VIEW_H_
#define _INSPECTOR_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QWidget>

// Project
#include "UI_GvvQInspectorView.h"

// GigaVoxels
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
class SampleCore;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class InspectorView : public QWidget, public Ui::GvvQInspectorView
{

	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * pParent ...
	 * pFlags ...
	 */
	InspectorView( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~InspectorView();

	/**
	 * Initialize this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void initialize( SampleCore* pPipeline );

	/**
	 * Populates this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void populate( SampleCore* pPipeline );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	GvCore::Array3D< uint >* _dataStructureChildArray;
	GvCore::Array3D< uint >* _dataStructureDataArray;

	GvCore::Array3DGPULinear< uint >* _nodeCacheTimeStampList;
	thrust::device_vector< uint >* _nodeCacheElementAddressList;
	uint _nodeCacheNbUnusedElements;

	GvCore::Array3DGPULinear< uint >* _brickCacheTimeStampList;
	thrust::device_vector< uint >* _brickCacheElementAddressList;
	uint _brickCacheNbUnusedElements;

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
	InspectorView( const InspectorView& );

	/**
	 * Copy operator forbidden.
	 */
	InspectorView& operator=( const InspectorView& );

	/********************************* SLOTS **********************************/

private slots:

};

#endif
