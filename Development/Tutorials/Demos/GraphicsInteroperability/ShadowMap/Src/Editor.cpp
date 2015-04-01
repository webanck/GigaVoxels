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

#include "Editor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "ShadowMap.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QString>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

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
 * Constructor
 *
 * pParent ...
 * pFlags ...
 ******************************************************************************/
Editor::Editor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_shadowMap( NULL )
,	_light( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Shadow Map Editor" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Editor::~Editor()
{
}

/******************************************************************************
 * Initialize this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void Editor::initialize( ShadowMap* pShadowMap, qglviewer::ManipulatedFrame* pLight )
{
	assert( pShadowMap != NULL );
	if ( pShadowMap != NULL )
	{
		_shadowMap = pShadowMap;
		_light = pLight;

		glm::vec3 tmp;

		tmp = _shadowMap->_cameraEye;
		_cameraEyeXDoubleSpinBox->setValue( tmp.x );
		_cameraEyeYDoubleSpinBox->setValue( tmp.y );
		_cameraEyeZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_cameraCenter;
		_cameraCenterXDoubleSpinBox->setValue( tmp.x );
		_cameraCenterYDoubleSpinBox->setValue( tmp.y );
		_cameraCenterZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_cameraUp;
		_cameraUpXDoubleSpinBox->setValue( tmp.x );
		_cameraUpYDoubleSpinBox->setValue( tmp.y );
		_cameraUpZDoubleSpinBox->setValue( tmp.z );
		_cameraFovYDoubleSpinBox->setValue( _shadowMap->_cameraFovY );
		_cameraAspectDoubleSpinBox->setValue( _shadowMap->_cameraAspectRatio );
		_cameraZNearDoubleSpinBox->setValue( _shadowMap->_cameraZNear );
		_cameraZFarDoubleSpinBox->setValue( _shadowMap->_cameraZFar );

		tmp = _shadowMap->_materialKa;
		_materialAmbientXDoubleSpinBox->setValue( tmp.x );
		_materialAmbientYDoubleSpinBox->setValue( tmp.y );
		_materialAmbientZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_materialKd;
		_materialDiffuseXDoubleSpinBox->setValue( tmp.x );
		_materialDiffuseYDoubleSpinBox->setValue( tmp.y );
		_materialDiffuseZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_materialKs;
		_materialSpecularXDoubleSpinBox->setValue( tmp.x );
		_materialSpecularYDoubleSpinBox->setValue( tmp.y );
		_materialSpecularZDoubleSpinBox->setValue( tmp.z );
		_materialShininessDoubleSpinBox->setValue( _shadowMap->_materialShininess );

		tmp = _shadowMap->_lightEye;
		_lightPositionXDoubleSpinBox->setValue( tmp.x );
		_lightPositionYDoubleSpinBox->setValue( tmp.y );
		_lightPositionZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_lightIntensity;
		_lightIntensityXDoubleSpinBox->setValue( tmp.x );
		_lightIntensityYDoubleSpinBox->setValue( tmp.y );
		_lightIntensityZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_lightCenter;
		_lightCenterXDoubleSpinBox->setValue( tmp.x );
		_lightCenterYDoubleSpinBox->setValue( tmp.y );
		_lightCenterZDoubleSpinBox->setValue( tmp.z );
		tmp = _shadowMap->_lightUp;
		_lightUpXDoubleSpinBox->setValue( tmp.x );
		_lightUpYDoubleSpinBox->setValue( tmp.y );
		_lightUpZDoubleSpinBox->setValue( tmp.z );
		_lightFovYDoubleSpinBox->setValue( _shadowMap->_lightFovY );
		_lightAspectDoubleSpinBox->setValue( _shadowMap->_lightAspectRatio );
		_lightZNearDoubleSpinBox->setValue( _shadowMap->_lightZNear );
		_lightZFarDoubleSpinBox->setValue( _shadowMap->_lightZFar );
	}
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraEyeXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_cameraEye;
	eye.x = pValue;
	_shadowMap->setCameraEye( eye );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraEyeYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_cameraEye;
	eye.y = pValue;
	_shadowMap->setCameraEye( eye );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraEyeZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_cameraEye;
	eye.z = pValue;
	_shadowMap->setCameraEye( eye );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraCenterXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_cameraCenter;
	center.x = pValue;
	_shadowMap->setCameraCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraCenterYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_cameraCenter;
	center.y = pValue;
	_shadowMap->setCameraCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraCenterZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_cameraCenter;
	center.z = pValue;
	_shadowMap->setCameraCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraUpXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_cameraUp;
	up.x = pValue;
	_shadowMap->setCameraUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraUpYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_cameraUp;
	up.y = pValue;
	_shadowMap->setCameraUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraUpZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_cameraUp;
	up.z = pValue;
	_shadowMap->setCameraUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraFovYDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setCameraFovY( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraAspectDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setCameraAspectRatio( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraZNearDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setCameraZNear( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__cameraZFarDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setCameraZFar( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialAmbientXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKa;
	material.x = pValue;
	_shadowMap->_materialKa = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialAmbientYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKa;
	material.y = pValue;
	_shadowMap->_materialKa = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialAmbientZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKa;
	material.z = pValue;
	_shadowMap->_materialKa = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialDiffuseXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKd;
	material.x = pValue;
	_shadowMap->_materialKd = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialDiffuseYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKd;
	material.y = pValue;
	_shadowMap->_materialKd = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialDiffuseZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKd;
	material.z = pValue;
	_shadowMap->_materialKd = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialSpecularXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKs;
	material.x = pValue;
	_shadowMap->_materialKs = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialSpecularYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKs;
	material.y = pValue;
	_shadowMap->_materialKs = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialSpecularZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 material = _shadowMap->_materialKs;
	material.z = pValue;
	_shadowMap->_materialKs = material;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__materialShininessDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->_materialShininess = pValue;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightPositionXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_lightEye;
	eye.x = pValue;
	_shadowMap->setLightEye( eye );

	_light->setPosition( eye.x, eye.y, eye.z );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightPositionYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_lightEye;
	eye.y = pValue;
	_shadowMap->setLightEye( eye );

	_light->setPosition( eye.x, eye.y, eye.z );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightPositionZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 eye = _shadowMap->_lightEye;
	eye.z = pValue;
	_shadowMap->setLightEye( eye );

	_light->setPosition( eye.x, eye.y, eye.z );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightIntensityXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 intensity = _shadowMap->_lightIntensity;
	intensity.x = pValue;
	_shadowMap->_lightIntensity = intensity;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightIntensityYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 intensity = _shadowMap->_lightIntensity;
	intensity.y = pValue;
	_shadowMap->_lightIntensity = intensity;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightIntensityZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 intensity = _shadowMap->_lightIntensity;
	intensity.z = pValue;
	_shadowMap->_lightIntensity = intensity;
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightCenterXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_lightCenter;
	center.x = pValue;
	_shadowMap->setLightCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightCenterYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_lightCenter;
	center.y = pValue;
	_shadowMap->setLightCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightCenterZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 center = _shadowMap->_lightCenter;
	center.z = pValue;
	_shadowMap->setLightCenter( center );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightUpXDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_lightUp;
	up.x = pValue;
	_shadowMap->setLightUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightUpYDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_lightUp;
	up.y = pValue;
	_shadowMap->setLightUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightUpZDoubleSpinBox_valueChanged( double pValue )
{
	glm::vec3 up = _shadowMap->_lightUp;
	up.z = pValue;
	_shadowMap->setLightUp( up );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightFovYDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setLightFovY( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightAspectDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setLightAspectRatio( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightZNearDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setLightZNear( pValue );
}

/******************************************************************************
 * Slot called when value has changed
 ******************************************************************************/
void Editor::on__lightZFarDoubleSpinBox_valueChanged( double pValue )
{
	_shadowMap->setLightZFar( pValue );
}
