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

#include "CustomSectionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineInterfaceViewer.h>
#include <GvvPipelineInterface.h>

// Project
#include "SampleCore.h"

// Qt
#include <QColorDialog>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

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
 * Default constructor
 ******************************************************************************/
CustomSectionEditor::CustomSectionEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
{
	setupUi( this );

	// Editor name
	setName( tr( "Infinite Sky" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void CustomSectionEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	SampleCore* pipeline = dynamic_cast< SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		_nbSpheresSpinBox->setValue( pipeline->getNbSpheres() );
        _nbSpheresTotalSpinBox->setValue( pipeline->getNbSpheresTotal() );
        //_minLevelToHandleUserDefinedRadioButton->setChecked( pipeline->getUserDefinedMinLevelOfResolutionToHandleMode() );

        // permet de definir a quel niveau on produit les spheres
        _minLevelToHandleUserDefinedSpinBox->setValue( pipeline->getUserDefinedMinLevelOfResolutionToHandle() );

        //_minLevelToHandleAutomaticRadioButton->setChecked( pipeline->getAutomaticMinLevelOfResolutionToHandleMode() );

        /* ------> A SUPPRIMER ?
        _minLevelToHandleAutomaticLineEdit->setText( QString::number( pipeline->getAutomaticMinLevelOfResolutionToHandle() ) );
        //*/

        _intersectionTypeComboBox->setCurrentIndex( pipeline->getSphereBrickIntersectionType() );
		_sphereRadiusFaderDoubleSpinBox->setValue( pipeline->getSphereRadiusFader() );
		_geometricCriteriaGroupBox->setChecked( pipeline->hasGeometricCriteria() );
		_minNbSpheresPerBrickSpinBox->setValue( pipeline->getMinNbSpheresPerBrick() );
		_screenBasedCriteriaGroupBox->setChecked( pipeline->hasScreenBasedCriteria() );
		_absoluteSizeCriteriaGroupBox->setChecked( pipeline->hasAbsoluteSizeCriteria() );
		_fixedSizeSphereRadiusRadioButton->setChecked( pipeline->hasFixedSizeSphere() );
		_fixedSizeSphereRadiusDoubleSpinBox->setValue( pipeline->getFixedSizeSphereRadius() );
        _sphereDiameterCoeffSpinBox->setValue( pipeline->getSphereDiameterCoeff() );

        _shaderScreenSpaceCoeffSpinBox->setValue( pipeline->getScreenSpaceCoeff() );

        /* ------> A SUPPRIMER ?
        _meanSizeOfSpheresRadioButton->setChecked( pipeline->hasMeanSizeOfSpheres() );
        //*/

        _shaderUseUniformColorCheckBox->setChecked( pipeline->hasShaderUniformColor() );
        _shaderUseAnimationCheckBox->setChecked( pipeline->hasShaderAnimation() );
        _shaderUseBlurChekBox->setChecked( pipeline->hasShaderBlurSphere() );
        _shaderUseFogChekBox->setChecked( pipeline->hasShaderFog() );
        _shaderFogDensitydoubleSpinBox->setValue( pipeline->getFogDensity() );
        _shaderUseLightSourceCheckBox->setChecked( pipeline->IsLightSourceType() );
        _shaderUseShadingCheckBox->setChecked( pipeline->hasShading() );
        _ShaderUseBugCorrectionCheckBox->setChecked( pipeline->hasBugCorrection() );

        _shaderIlluminationCoeffDoubleSpinBox->setValue( pipeline->getIlluminationCoeff() );
    }
}

/******************************************************************************
 * Slot called when global nb spheres value has changed
 ******************************************************************************/
void CustomSectionEditor::on__nbSpheresSpinBox_valueChanged( int pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setNbSpheres( pValue );

    // Update widget
    //_minLevelToHandleAutomaticLineEdit->setText( QString::number( sampleCore->getAutomaticMinLevelOfResolutionToHandle() ) );
}

/******************************************************************************
 * Slot called when global nb spheres value has changed
 ******************************************************************************/
void CustomSectionEditor::on__nbSpheresTotalSpinBox_valueChanged( int pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setNbSpheresTotal( pValue );

    // Update widget
    //_minLevelToHandleAutomaticLineEdit->setText( QString::number( sampleCore->getAutomaticMinLevelOfResolutionToHandle() ) );
}

/******************************************************************************
 * Slot called when regenerate positions button is pressed
 ******************************************************************************/
void CustomSectionEditor::on__regeneratePositions_pressed()
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->regeneratePositions();
}

/******************************************************************************
 * Slot called when regenerate positions button is pressed
 ******************************************************************************/
void CustomSectionEditor::on__regeneratePositionsBis_pressed()
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->regeneratePositions();
}

/******************************************************************************
 * Slot called when user defined min level of resolution to handle radio button is toggled
 ******************************************************************************/
void CustomSectionEditor::on__minLevelToHandleUserDefinedRadioButton_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setUserDefinedMinLevelOfResolutionToHandleMode( pChecked );
}

/******************************************************************************
 * Slot called when user defined min level of resolution to handle value has changed
 ******************************************************************************/
void CustomSectionEditor::on__minLevelToHandleUserDefinedSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setUserDefinedMinLevelOfResolutionToHandle( pValue );

	// Update widget
    //_minLevelToHandleAutomaticLineEdit->setText( QString::number( sampleCore->getAutomaticMinLevelOfResolutionToHandle() ) );
}

/******************************************************************************
 * Slot called when automatic min level of resolution to handle radio button is toggled
 ******************************************************************************/
void CustomSectionEditor::on__minLevelToHandleAutomaticRadioButton_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setAutomaticMinLevelOfResolutionToHandleMode( pChecked );
}

/******************************************************************************
 * Slot called when intersection type value has changed
 ******************************************************************************/
void CustomSectionEditor::on__intersectionTypeComboBox_currentIndexChanged( int pIndex )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setSphereBrickIntersectionType( pIndex );
}

/******************************************************************************
 * Slot called when global sphere radius fader value has changed
 ******************************************************************************/
void CustomSectionEditor::on__sphereRadiusFaderDoubleSpinBox_valueChanged( double pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setSphereRadiusFader( pValue );
}

/******************************************************************************
 * Slot called when the geometric criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__geometricCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setGeometricCriteria( pChecked );
}

/******************************************************************************
 * Slot called when min nb of spheres per brick value has changed
 ******************************************************************************/
void CustomSectionEditor::on__minNbSpheresPerBrickSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setMinNbSpheresPerBrick( pValue );
}

/******************************************************************************
 * Slot called when the screen based criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__screenBasedCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setScreenBasedCriteria( pChecked );
}

/******************************************************************************
 * Slot called when the absolute size criteria group box state has changed
 ******************************************************************************/
void CustomSectionEditor::on__absoluteSizeCriteriaGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setAbsoluteSizeCriteria( pChecked );
}

/******************************************************************************
 * Slot called when fixed size of spheres radio button is toggled
 ******************************************************************************/
void CustomSectionEditor::on__fixedSizeSphereRadiusRadioButton_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setFixedSizeSphere( pChecked );
}

/******************************************************************************
 * Slot called when fixed size radius of spheres value has changed
 ******************************************************************************/
void CustomSectionEditor::on__fixedSizeSphereRadiusDoubleSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setFixedSizeSphereRadius( pValue );
}

/******************************************************************************
 * Slot called when sphere diameter coefficient value has changed
 ******************************************************************************/
void CustomSectionEditor::on__sphereDiameterCoeffSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setSphereDiameterCoeff( pValue );
}


/******************************************************************************
 * Slot called when sphere diameter coefficient value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderScreenSpaceCoeffSpinBox_valueChanged( int pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setScreenSpaceCoeff( pValue );
}


/******************************************************************************
 * Slot called when mean size of spheres radio button is toggled
 ******************************************************************************/
void CustomSectionEditor::on__meanSizeOfSpheresRadioButton_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setMeanSizeOfSpheres( pChecked );
}

/******************************************************************************
 * Slot called when shader's uniform color check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseUniformColorCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	sampleCore->setShaderUniformColorMode( pChecked );
}

/******************************************************************************
 * Slot called when shader's uniform color tool button is released
 ******************************************************************************/
void CustomSectionEditor::on__shaderUniformColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );
			if ( sampleCore != NULL )
			{
				sampleCore->setShaderUniformColor( static_cast< float >( color.red() ) / 255.f
												, static_cast< float >( color.green() ) / 255.f
												, static_cast< float >( color.blue() ) / 255.f
												, static_cast< float >( color.alpha() ) / 255.f );
			}
		}
	}
}

/******************************************************************************
 * Slot called when shader's animation check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseAnimationCheckBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setShaderAnimation( pChecked );
}

/******************************************************************************
 * Slot called when shader's blur sphere check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseBlurChekBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setShaderBlurSphere( pChecked );
}

/******************************************************************************
 * Slot called when shader's fog check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseFogChekBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setShaderFog( pChecked );
}

/******************************************************************************
 * Slot called when fog density value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderFogDensitydoubleSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setFogDensity( pValue );
}


/******************************************************************************
 * Slot called when shader's fog color tool button is released
 ******************************************************************************/
void CustomSectionEditor::on__shaderFogColorToolButton_released()
{
    QColor color = QColorDialog::getColor( Qt::white, this );
    if ( color.isValid() )
    {
        GvvApplication& application = GvvApplication::get();
        GvvMainWindow* mainWindow = application.getMainWindow();
        Gvv3DWindow* window3D = mainWindow->get3DWindow();
        GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
        if ( pipelineViewer != NULL )
        {
            GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

            SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
            assert( sampleCore != NULL );
            if ( sampleCore != NULL )
            {
                sampleCore->setShaderFogColor(  static_cast< float >( color.red() ) / 255.f,
                                                static_cast< float >( color.green() ) / 255.f,
                                                static_cast< float >( color.blue() ) / 255.f,
                                                static_cast< float >( color.alpha() ) / 255.f
                                                );
            }
        }
    }
}

/******************************************************************************
 * Slot called when shader's light source check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseLightSourceCheckBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setLightSourceType( pChecked );
}



/******************************************************************************
 * Slot called when shading check box is toggled
 ******************************************************************************/
void CustomSectionEditor::on__shaderUseShadingCheckBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setShading( pChecked );
}


/******************************************************************************
 * Slot called when shading bug correction is toggled
 ******************************************************************************/
void CustomSectionEditor::on__ShaderUseBugCorrectionCheckBox_toggled( bool pChecked )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setBugCorrection( pChecked );
}

/******************************************************************************
 * Slot called when fog density value has changed
 ******************************************************************************/
void CustomSectionEditor::on__shaderIlluminationCoeffDoubleSpinBox_valueChanged( double pValue )
{
    GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

    sampleCore->setIlluminationCoeff( pValue );
}

/******************************************************************************
 * Slot called when number of reflection value is changed
 ******************************************************************************/
void CustomSectionEditor::on__numberOfReflectionsSpinBox_valueChanged( int pValue )
{
	GvvApplication& application = GvvApplication::get();
    GvvMainWindow* mainWindow = application.getMainWindow();
    Gvv3DWindow* window3D = mainWindow->get3DWindow();
    GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	
    GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

    SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
    assert( sampleCore != NULL );

	sampleCore->setNumberOfReflections( pValue );
}