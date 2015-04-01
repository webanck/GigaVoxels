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

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/


/**
 * This file defines all the events that may be used to monitor an application
 */

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( cpmApplicationDefaultFrameEvent )	/* Frame event, should not be removed */

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( application_userEvt0 )

/**
 * Main frame
 */
CUDAPM_DEFINE_EVENT( frame )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( app_init_frame )
CUDAPM_DEFINE_EVENT( app_post_frame )

/**
 * Copy data structure from device to host
 */
CUDAPM_DEFINE_EVENT( copy_dataStructure_to_host )

/**
 * Data Production Manager - Clear cache
 */
CUDAPM_DEFINE_EVENT( gpucache_clear )

/**
 * Cache Manager - clear cache
 */
CUDAPM_DEFINE_EVENT( gpucachemgr_clear_cpyAddr ) // not used...
CUDAPM_DEFINE_EVENT( gpucachemgr_clear_fillML )	// Temp masks
CUDAPM_DEFINE_EVENT( gpucachemgr_clear_fillTimeStamp ) // Timestamps

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_preRenderPass )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( vsrender_init_frame )  // not used...
CUDAPM_DEFINE_EVENT( vsrender_copyconsts_frame )

/**
 * Main rendering stage
 */
CUDAPM_DEFINE_EVENT( gv_rendering )
CUDAPM_DEFINE_EVENT( vsrender_initRays ) // not used...
CUDAPM_DEFINE_EVENT( vsrender_endRays )

/**
 * Data Production Management
 */
CUDAPM_DEFINE_EVENT( dataProduction_handleRequests )

/**
 * Manage requests
 */
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests )
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests_createMask ) // not used...
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests_elemsReduction )
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests_my_copy_if_0 ) // not used...
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests_my_copy_if_1 ) // not used...
CUDAPM_DEFINE_EVENT( dataProduction_manageRequests_my_copy_if_2 ) // not used...

/**
 * Update timestamps
 */
CUDAPM_DEFINE_EVENT( cache_updateTimestamps )
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_dataStructure )
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_bricks )
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_createMasks )
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_threadReduc ) // not used...
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_threadReduc1 )
CUDAPM_DEFINE_EVENT( cache_updateTimestamps_threadReduc2 )

/**
 * Production
 */
CUDAPM_DEFINE_EVENT( producer_nodes )
CUDAPM_DEFINE_EVENT( producer_bricks )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_updateSymbols )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucachemgr_createUpdateList_createMask )
CUDAPM_DEFINE_EVENT( gpucachemgr_createUpdateList_elemsReduction )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_nodes_manageUpdates )
CUDAPM_DEFINE_EVENT( gpucache_nodes_createMask )
CUDAPM_DEFINE_EVENT( gpucache_nodes_elemsReduction )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_nodes_subdivKernel )
CUDAPM_DEFINE_EVENT( gpucache_nodes_preLoadMgt )
CUDAPM_DEFINE_EVENT( gpucache_nodes_preLoadMgt_gpuProd )
CUDAPM_DEFINE_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )
CUDAPM_DEFINE_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_bricks_bricksInvalidation )
CUDAPM_DEFINE_EVENT( gpucache_bricks_copyTransfer )
CUDAPM_DEFINE_EVENT( gpucache_bricks_cpuSortRequests )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_bricks_manageUpdates )
CUDAPM_DEFINE_EVENT( gpucache_bricks_createMask )
CUDAPM_DEFINE_EVENT( gpucache_bricks_elemsReduction )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_bricks_getLocalizationInfo )
CUDAPM_DEFINE_EVENT( gpucache_bricks_gpuFetchBricks )
CUDAPM_DEFINE_EVENT( gpucache_bricks_gpuFetchBricks_constUL )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucache_bricks_loadBricks )
CUDAPM_DEFINE_EVENT( gpucache_bricks_manageConsts )
CUDAPM_DEFINE_EVENT( gpucache_bricks_updateOctreeBricks )
CUDAPM_DEFINE_EVENT( gpucache_bricks_preLoadMgt )
CUDAPM_DEFINE_EVENT( gpucache_bricks_preLoadMgt_gpuProd )
CUDAPM_DEFINE_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )
CUDAPM_DEFINE_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
CUDAPM_DEFINE_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )
CUDAPM_DEFINE_EVENT( copyToTextureTest0 )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( gpucachemgr_updateSymbols )
CUDAPM_DEFINE_EVENT( gpucachemgr_updateTimeStampsCPU )

/**
 * ...
 */
CUDAPM_DEFINE_EVENT( cudakernelRenderGigaVoxels )
CUDAPM_DEFINE_EVENT( cudainternalRenderGigaVoxels )


CUDAPM_DEFINE_EVENT( gpucache_update_VBO )

		
CUDAPM_DEFINE_EVENT( gpucache_update_VBO_createMask )
CUDAPM_DEFINE_EVENT( gpucache_update_VBO_compaction )
CUDAPM_DEFINE_EVENT( gpucache_update_VBO_nb_pts )
CUDAPM_DEFINE_EVENT( gpucache_update_VBO_parallel_prefix_sum )
CUDAPM_DEFINE_EVENT( gpucache_update_VBO_update_VBO )

/**
 * Pre/Post frame
 * Map/unmap graphics resources
 */
CUDAPM_DEFINE_EVENT( vsrender_pre_frame )
CUDAPM_DEFINE_EVENT( vsrender_pre_frame_mapbuffers )
CUDAPM_DEFINE_EVENT( vsrender_post_frame )
CUDAPM_DEFINE_EVENT( vsrender_post_frame_unmapbuffers )

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
