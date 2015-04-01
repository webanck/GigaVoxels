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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QApplication>
#include <QFileInfo>
#include <QDir>

// Project
#include "GvxVoxelizerDialog.h"
#include "GvxVoxelizerEngine.h"
#include "GvxDataTypeHandler.h"
#include "GvxAssimpSceneVoxelizer.h"

// STL
#include <string>
#include <iostream>
#include <cassert>
#include <sstream>

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// TinyXML
#include <tinyxml.h>
//#include <tinystr.h>


// TO DO
// This CImg dependency should be placed in an encapsulated class...
// ...
// CImg
#define cimg_use_magick	// Beware, this definition must be placed before including CImg.h
#include <CImg.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Project
using namespace Gvx;

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

// CImg
void printCImgLibraryInfo();

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char* pArgv[] )
{
	// Exit code
	int result = 0;

	// LOG
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "-------- GigaVoxels Voxelizer --------" << std::endl;
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "\n" << std::endl;
	// TO DO
	// - add a version
	// ...

#ifndef NDEBUG
	// LOG : print CImg settings
	printCImgLibraryInfo();
#endif

	// Qt main application
	QApplication application( pArgc, pArgv );
	
	// Show a USER-dialog to parameterize the voxelization process
	GvxVoxelizerDialog voxelizerDialog;
	if ( QDialog::Rejected == voxelizerDialog.exec() )
	{
		// Handle error
		return 1;
	}

	// Create a scene voxelizer
	GvxSceneVoxelizer* sceneVoxelizer = new GvxAssimpSceneVoxelizer();
	if ( sceneVoxelizer == NULL )
	{
		// TO DO
		// Handle error
		// ...
		return 2;
	}

	// Initialize the scene voxelizer
	QFileInfo fileInfo( voxelizerDialog._fileName );
	sceneVoxelizer->setFilePath( QString( fileInfo.absolutePath() + QDir::separator() ).toLatin1().constData() );
	sceneVoxelizer->setFileName( fileInfo.completeBaseName().toLatin1().constData() );
	sceneVoxelizer->setFileExtension( QString( "." + fileInfo.suffix() ).toLatin1().constData() );
	sceneVoxelizer->setMaxResolution( voxelizerDialog._maxResolution );
	sceneVoxelizer->setBrickWidth( voxelizerDialog._brickWidth );
	sceneVoxelizer->setDataType( static_cast< Gvx::GvxDataTypeHandler::VoxelDataType >( voxelizerDialog._dataType ) );
	sceneVoxelizer->setFilterType(voxelizerDialog._filterType);
	sceneVoxelizer->setFilterIterations(voxelizerDialog._nbFilterOperation);
	sceneVoxelizer->setNormals(voxelizerDialog._normals);

	// TO DO
	// Check input data here or in the voxelizer.
	// Because, in the future, the voxelizer could be called either by a GUI interface or by command line (with a configuration batch file)
	//...
	
	// LOG
	std::cout << "-------- BEGIN voxelization process --------" << std::endl;
	std::cout << "\n" << std::endl;

	// Launch the voxelization
	sceneVoxelizer->launchVoxelizationProcess();

	// LOG
	std::cout << "\n" << std::endl;
	std::cout << "-------- END voxelization process --------" << std::endl;
		

	// Enter Qt main event loop
	//result = application.exec();


	int nbChannels;
	if (voxelizerDialog._normals)
	{
		nbChannels=2;
	} else {
		nbChannels=1;
	}



	TiXmlDocument doc;
	TiXmlElement * root;
	root = new TiXmlElement( "Model" );  
    root->SetAttribute("name",sceneVoxelizer->getFileName().c_str());
    root->SetAttribute("directory",".");

	stringstream nbLevels;
	nbLevels<<(sceneVoxelizer->getMaxResolution())+1;

	root->SetAttribute("nbLevels",nbLevels.str().c_str());


	printf("%d %d\n",sceneVoxelizer->getBrickWidth(),sceneVoxelizer->getMaxResolution());

	TiXmlElement * tree = new TiXmlElement( "NodeTree" );
	TiXmlElement * level;
	TiXmlElement * channel;

	for (int k = 0;k<(sceneVoxelizer->getMaxResolution())+1;k++)
	{
		level=  new TiXmlElement( "Level" );
		stringstream levelK;
		levelK<<k;
		level->SetAttribute("id",levelK.str().c_str());
		stringstream filename;
		filename<<sceneVoxelizer->getFileName()<<"_BR"<<sceneVoxelizer->getBrickWidth()<<"_B1_L"<<k<<".nodes";
		level->SetAttribute("filename",filename.str().c_str());
		tree->LinkEndChild(level);
	}
	root->LinkEndChild(tree);

	tree=  new TiXmlElement( "BrickData" );
	stringstream brickRes;
	brickRes<<sceneVoxelizer->getBrickWidth();
	tree->SetAttribute("brickResolution",brickRes.str().c_str());
	tree->SetAttribute("borderSize","1");

	for (int p = 0;p<nbChannels;p++)
	{
		channel=  new TiXmlElement( "Channel" );
		stringstream channelP;
		channelP<<p;
		channel->SetAttribute("id",channelP.str().c_str());
		if (p==0) 
		{

			channel->SetAttribute("name","color");
			channel->SetAttribute("type","uchar4");
		} else if (p==1){
			channel->SetAttribute("name","normal");
			channel->SetAttribute("type","half4");
		} else {
			channel->SetAttribute("name","unknown");
			channel->SetAttribute("type","unknown");
		}

		for (int k = 0;k<(sceneVoxelizer->getMaxResolution())+1;k++)
		{
			level=  new TiXmlElement( "Level" );
			stringstream levelK;
			levelK<<k;
			level->SetAttribute("id",levelK.str().c_str());

			stringstream filename;
			if (p==0) 
			{
				filename<<sceneVoxelizer->getFileName()<<"_BR"<<sceneVoxelizer->getBrickWidth()<<"_B1_L"<<k<<"_C"<<p<<"_uchar4"<<".bricks";
			} else if (p==1) {
				filename<<sceneVoxelizer->getFileName()<<"_BR"<<sceneVoxelizer->getBrickWidth()<<"_B1_L"<<k<<"_C"<<p<<"_half4"<<".bricks";
			} else {
				filename<<sceneVoxelizer->getFileName()<<"_BR"<<sceneVoxelizer->getBrickWidth()<<"_B1_L"<<k<<"_C"<<p<<"_unknown"<<".bricks";
			}

			level->SetAttribute("filename",filename.str().c_str());

			channel->LinkEndChild(level);
		}



		tree->LinkEndChild(channel);
	}

	root->LinkEndChild(tree);
    doc.LinkEndChild( root );

	stringstream XMLFilename;
	XMLFilename<<sceneVoxelizer->getFileName()<<".xml";

    doc.SaveFile( XMLFilename.str().c_str() );

	// Return exit code
	return result;
}

/******************************************************************************
 * Print informations about the CImg library environement variables.
 *
 * TO DO : move this method in a CImg wrapper
 ******************************************************************************/
void printCImgLibraryInfo()
{
	cimg_library::cimg::info();
}
