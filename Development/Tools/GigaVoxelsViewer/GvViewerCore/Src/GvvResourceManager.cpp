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
//
//#include "GvvPipelineManager.h"
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// GvViewer
//#include "GvvPipelineInterface.h"
//#include "GvvPipelineManagerListener.h"
//
//// System
//#include <cassert>
//#include <cstdio>
//#include <cstdlib>
//#include <iostream>
//#include <cstring>
//
//// STL
//#include <algorithm>
//
///******************************************************************************
// ****************************** NAMESPACE SECTION *****************************
// ******************************************************************************/
//
//// GvViewer
//using namespace GvViewerCore;
//
//// STL
//using namespace std;
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///**
// * The unique instance of the singleton.
// */
//GvvPipelineManager* GvvPipelineManager::msInstance = NULL;
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** METHOD DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// * Get the unique instance.
// *
// * @return the unique instance
// ******************************************************************************/
//GvvPipelineManager& GvvPipelineManager::get()
//{
//    if ( msInstance == NULL )
//    {
//        msInstance = new GvvPipelineManager();
//    }
//
//    return *msInstance;
//}
//
///******************************************************************************
// * Constructor
// ******************************************************************************/
//GvvPipelineManager::GvvPipelineManager()
//:	mPipelines()
//,	mListeners()
//{
//}
//
///******************************************************************************
// * Add a pipeline.
// *
// * @param the pipeline to add
// ******************************************************************************/
//void GvvPipelineManager::addPipeline( GvvPipelineInterface* pPipeline )
//{
//	assert( pPipeline != NULL );
//	if ( pPipeline != NULL )
//	{
//		// Add pipeline
//		mPipelines.push_back( pPipeline );
//
//		// Inform listeners that a pipeline has been added
//		vector< GvvPipelineManagerListener* >::iterator it = mListeners.begin();
//		for ( ; it != mListeners.end(); ++it )
//		{
//			GvvPipelineManagerListener* listener = *it;
//			if ( listener != NULL )
//			{
//				listener->onPipelineAdded( pPipeline );
//			}
//		}
//	}
//}
//
///******************************************************************************
// * Remove a pipeline.
// *
// * @param the pipeline to remove
// ******************************************************************************/
//void GvvPipelineManager::removePipeline( GvvPipelineInterface* pPipeline )
//{
//	assert( pPipeline != NULL );
//	if ( pPipeline != NULL )
//	{
//		vector< GvvPipelineInterface * >::iterator itPipeline;
//		itPipeline = find( mPipelines.begin(), mPipelines.end(), pPipeline );
//		if ( itPipeline != mPipelines.end() )
//		{
//			// Remove pipeline
//			mPipelines.erase( itPipeline );
//
//			// Inform listeners that a pipeline has been removed
//			vector< GvvPipelineManagerListener* >::iterator itListener = mListeners.begin();
//			for ( ; itListener != mListeners.end(); ++itListener )
//			{
//				GvvPipelineManagerListener* listener = *itListener;
//				if ( listener != NULL )
//				{
//					listener->onPipelineAdded( pPipeline );
//				}
//			}
//		}
//	}
//}
//
///******************************************************************************
// * Register a listener.
// *
// * @param pListener the listener to register
// ******************************************************************************/
//void GvvPipelineManager::registerListener( GvvPipelineManagerListener* pListener )
//{
//	assert( pListener != NULL );
//	if ( pListener != NULL )
//	{
//		// Add listener
//		mListeners.push_back( pListener );
//	}
//}
//
///******************************************************************************
// * Unregister a listener.
// *
// * @param pListener the listener to unregister
// ******************************************************************************/
//void GvvPipelineManager::unregisterListener( GvvPipelineManagerListener* pListener )
//{
//	assert( pListener != NULL );
//	if ( pListener != NULL )
//	{
//		vector< GvvPipelineManagerListener * >::iterator it;
//		it = find( mListeners.begin(), mListeners.end(), pListener );
//		if ( it != mListeners.end() )
//		{
//			// Remove pipeline
//			mListeners.erase( it );
//		}
//	}
//}
