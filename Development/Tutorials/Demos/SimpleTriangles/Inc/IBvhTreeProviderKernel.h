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

#ifndef _I_BVH_TREE_PROVIDER_KERNEL_H_
#define _I_BVH_TREE_PROVIDER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace gigavoxels
{

	/** 
	 * @class IBvhTreeProviderKernel
	 *
	 * @brief The IBvhTreeProviderKernel class provides...
	 *
	 * ...
	 */
	template< uint Id, typename Derived >
	class IBvhTreeProviderKernel
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param d ...
		 */
		IBvhTreeProviderKernel( Derived& d )
		:	mDerived( d )
		{
		}

		/**
		 * ...
		 *
		 * @param gpuPool ...
		 * @param requestID ...
		 * @param processID ...
		 * @param newElemAddress ...
		 * @param parentLocInfo ...
		 *
		 * @return ...
		 */
		template< typename GPUPoolKernelType >
		__device__
		inline uint produceData( GPUPoolKernelType& gpuPool, uint requestID, uint processID,
								uint3 newElemAddress, VolTreeBVHNodeUser& node )
		{
			return mDerived.produceData( gpuPool, requestID, processID, newElemAddress, node, Loki::Int2Type< Id >() );
		}

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/******************************* ATTRIBUTES *******************************/

		/**
		 * ...
		 */
		Derived mDerived;

		/******************************** METHODS *********************************/
				
	};

} // namespace gigavoxels

#endif // !_I_BVH_TREE_PROVIDER_KERNEL_H_
