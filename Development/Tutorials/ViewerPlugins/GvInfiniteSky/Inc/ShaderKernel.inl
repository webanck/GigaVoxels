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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * This method is called just before the cast of a ray. Use it to initialize any data
 *  you may need. You may also want to modify the initial distance along the ray (tTree).
 *
 * @param pRayStartTree the starting position of the ray in octree's space.
 * @param pRayDirTree the direction of the ray in octree's space.
 * @param pTTree the distance along the ray's direction we start from.
 ******************************************************************************/
__device__
inline void ShaderKernel::preShadeImpl( const float3& pRayStartTree, const float3& pRayDirTree, float& pTTree )
{
    _accColor = make_float3( 0.f );
    _accTransparency = 1.f;
    _accFogDepth = 0.f;
	_renderViewContext = k_renderViewContext.viewCenterTP;
	_distanceBeforeReflection = 0.f;
	
    //_finalColor = make_float4( 0.f );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
__device__
inline void ShaderKernel::postShadeImpl( /*int pCounter*/ )
{
    /*
    if ( _accColor.w >= cOpacityThreshold )
    {
        _accColor.w = 1.f;
    }
    */
    if ( 1 - _accTransparency >= cOpacityThreshold )
    {
        _accTransparency = 0.f;
    }
}

/******************************************************************************
 * This method returns the cone aperture for a given distance.
 *
 * @param pTTree the current distance along the ray's direction.
 *
 * @return the cone aperture
 ******************************************************************************/
__device__
inline float ShaderKernel::getConeApertureImpl( const float pTTree ) const
{
	// Overestimate to avoid aliasing
	const float scaleFactor = 1.333f;

	// It is an estimation of the size of a voxel at given distance from the camera.
	// It is based on THALES theorem. Its computation is rotation invariant.
	return k_renderViewContext.pixelSize.x * pTTree * ( scaleFactor * k_renderViewContext.frustumNearINV );
}

/******************************************************************************
 * This method returns the final rgba color that will be written to the color buffer.
 *
 * @return the final rgba color.
 ******************************************************************************/
__device__
inline float4 ShaderKernel::getColorImpl() const
{
    return make_float4(_accColor.x, _accColor.y, _accColor.z, 1 - _accTransparency);
    //return _finalColor;
}

/******************************************************************************
 * This method is called before each sampling to check whether or not the ray should stop.
 *
 * @param pRayPosInWorld the current ray's position in world space.
 *
 * @return true if you want to continue the ray. false otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::stopCriterionImpl( const float3& pRayPosInWorld ) const
{
    //return ( _accColor.w >= cOpacityThreshold );
    if( 1 - _accTransparency >= cOpacityThreshold ){
        //_finalColor = make_float4(_accColor.x, _accColor.y, _accColor.z, 1.0);// *_accOpacity;
        return true;
    }
    return false;
}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.

 * @param pElementSize the desired element size in the current octree level.
 *
 * @param pConeAperture the ConeAperture at the considered point
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::descentCriterionImpl( const float pElementSize, const float pConeAperture ) const
{
	if ( cScreenBasedCriteria )
	{
		if ( pElementSize < pConeAperture * cScreenSpaceCoeff )
		{
			return false;
		}
	}
	return true;
}

/******************************************************************************
 * This method is called for each sample. For example, shading or secondary rays
 * should be done here.
 *
 * @param pBrickSampler brick sampler
 * @param pSamplePosScene position of the sample in the scene
 * @param pRayDir ray direction
 * @param pRayStep ray step
 * @param pConeAperture cone aperture
 ******************************************************************************/
template< typename SamplerType >
__device__
inline void ShaderKernel::runImpl( const SamplerType& pBrickSampler, const float3 pSamplePosScene, const float3 pRayDir, float& pRayStep, const float pConeAperture )
{
	// Here, we are in a node and we know that there is at least one sphere
	// due to the producer's oracle.

	// brickInfo contient la dimension de la brique dans les 3 premiers parametres puis le nombre de spheres dans la brique en 4eme parametre
	const float4 brickInfo = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																0.5f * pBrickSampler._volumeTree->brickCacheResINV );

	// brickGeoData contient la position de la brique dans le cube gigavoxel [0; 1] puis la taille de la brique
	const float4 brickData = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																0.5f * pBrickSampler._volumeTree->brickCacheResINV + make_float3( pBrickSampler._volumeTree->brickCacheResINV.x, 0.f, 0.f ) );
	
	const float3 eyeToBrickVector = make_float3( brickData.x, brickData.y, brickData.z ) - _renderViewContext;


    //bool test = true;

	// Iterate through spheres
	for ( int i = 2; i < brickInfo.w + 2 ; ++i )
	{

       // Retrieve sphere index
		uint3 index3D;
		index3D.x = i % static_cast< uint >( brickInfo.x );
		index3D.y = ( i / static_cast< uint >( brickInfo.x ) ) % static_cast< uint >( brickInfo.y );
		index3D.z = i / static_cast< uint >( brickInfo.x * brickInfo.y );
		
		// Sample data structrure to retrieve sphere data (position and radius)
        const float4 sphereData = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >
																	( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																	0.5f * pBrickSampler._volumeTree->brickCacheResINV +  make_float3( index3D.x * pBrickSampler._volumeTree->brickCacheResINV.x,
																	index3D.y * pBrickSampler._volumeTree->brickCacheResINV.y,
																	index3D.z * pBrickSampler._volumeTree->brickCacheResINV.z) );
		
        float sphereRadius = sphereData.w;
		float3 spherePosition = make_float3( sphereData.x, sphereData.y, sphereData.z );

		// Animation
		if ( cShaderAnimation )
		{
            // Animate sphere radius
            sphereRadius = sphereRadius - /*scale*/0.25f * ( 0.5f * ( 1.0f + cosf( 2 * 3.141592f * 0.04f * k_currentTime + /* phase */8324.17f * sphereData.w - 7754.33f/* * i*/) ) ) * sphereData.w;

			// Animate sphere position
			//spherePosition = make_float3( spherePosition.x + brickData.x, spherePosition.y + brickData.y, spherePosition.z + brickData.z );
			//spherePosition = spherePosition - /*scale*/0.25f * ( 0.5f * ( 1.0f + cosf( 2 * 3.141592f * 0.004f * k_currentTime + /* phase */7.0f * sphereData.x + /* phase */823.0f * sphereData.y - /* phase */100.0f * sphereData.z ) ) ) * sphereData.w;
			// TO TEST :
			// Utiliser poids faible de "x"
		}

		// Sphere ray-Tracing with anti-aliasing

//
//                 __ __ __ __ __ __ __ __ 
//                |                       |
//                |            (sphere 1) |
//                |               X       |
//   (eye)        |               |  D    |             Ray
//    X---->------|---------------|-------|--------------->
//     -->        |                       |
//     d          |       (sphere 2)      |
//                |            X          |
//                |__ __ __ __ __ __ __ __|
//                X 
//                (brick position)
//

		const float3 eyeToSphere = eyeToBrickVector + spherePosition;
		const float distanceFromEyeToProjectedSphereOnRay = dot( eyeToSphere, pRayDir );
		const float coneAperture = getConeAperture( _distanceBeforeReflection + distanceFromEyeToProjectedSphereOnRay );

		// Compute distance D between sphere center and the ray
		//const float D = sqrtf( max( 0.f , dot( eyeToSphere, eyeToSphere ) - distanceFromEyeToProjectedSphereOnRay * distanceFromEyeToProjectedSphereOnRay ) );
        const float D = sqrtf(  max( 0.f ,dot( eyeToSphere, eyeToSphere ) - distanceFromEyeToProjectedSphereOnRay * distanceFromEyeToProjectedSphereOnRay /*+ 1e-6f*/)/*in case of float error precision*/ );
	
		// Cone aperture
		const float Rv = coneAperture * 0.5f;



        // color opacity to represent anti-aliasing or blur
        float objectTransparency = 0.0f;
        float antiAliasingTransparency = 0.0f;

        // Handle ray-sphere intersection
        if ( D - Rv - sphereRadius < 0.0f )
        {

            bool withoutBugCorrection = true;
            float lambda = 0.0;

/******** BUG AFFICHAGE AVEC QUADRILLAGE */
            if( cShaderBugCorrection )
            {

                float distEyeToSphere  = sqrtf(eyeToSphere.x*eyeToSphere.x +
                                               eyeToSphere.y*eyeToSphere.y +
                                               eyeToSphere.z*eyeToSphere.z
                                               );

                // distance from eye to the first point of the sphere
                lambda =  distEyeToSphere * (
                                                    1 - sqrtf(
                                                                (distanceFromEyeToProjectedSphereOnRay*distanceFromEyeToProjectedSphereOnRay)/(distEyeToSphere*distEyeToSphere) -
                                                                (distEyeToSphere*distEyeToSphere - sphereRadius*sphereRadius)/(distEyeToSphere*distEyeToSphere)
                                                            )
                                                    );
    /*
                float3 eyeToBrickIntersection = pSamplePosScene - k_renderViewContext.viewCenterTP;
                //float distToBrickIntersection = length(eyeToBrickIntersection);

                float distToBrickIntersection  = sqrtf(eyeToBrickIntersection.x*eyeToBrickIntersection.x +
                                                       eyeToBrickIntersection.y*eyeToBrickIntersection.y +
                                                       eyeToBrickIntersection.z*eyeToBrickIntersection.z
                                                       );
    //*/

                float3 p = _renderViewContext + lambda * pRayDir;
    //          float dist_p = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);

                if( threadIdx.x == 2 && threadIdx.y == 0 && threadIdx.z == 0 )
                {
                    /*
                    printf("Position P = (%f %f %f)\ndist P = %f\nlambda = %f\ndist to brick = %f\ndist eye to sphere = %f\nsphereRadius = %f\n----------------------\n",
                           p.x, p.y, p.z,
                           dist_p,
                           lambda,
                           distToBrickIntersection,
                           distEyeToSphere,sphereRadius);
                    //*/
                }

                // the point isn't into the brick --> we don't color the sphere
                if( ! (p.x >= brickData.x && p.x < brickData.x + brickData.w &&
                       p.y >= brickData.y && p.y < brickData.y + brickData.w &&
                       p.z >= brickData.z && p.z < brickData.z + brickData.w)
                  )
                {
                    withoutBugCorrection = false;
                }
            }

/******************************/

            //if( distToBrickIntersection < lambda ){
            if( withoutBugCorrection )
            {

                // sphere color to accumulate
                float3 color;

                if ( cShaderUseUniformColor )
                {
                    color = make_float3(cShaderUniformColor.x, cShaderUniformColor.y, cShaderUniformColor.z);
                }
                else color = make_float3( ( brickData.x + sphereData.x ), ( brickData.y + sphereData.y ), ( brickData.z + sphereData.z ));


                color *= cSphereIlluminationCoeff;


                //***** Anti-aliasing *************************************************

                // Sphere is entirely in the voxel (very small)
                if ( D + sphereRadius < Rv )
                {
                    // Evaluate ratio surface of 2D discs ( pi * r² )
                    antiAliasingTransparency = 1 - ( sphereRadius * sphereRadius ) / ( Rv * Rv );
                }
                // Voxel is entirely in the sphere
                else if ( D + Rv < sphereRadius )
                {
                    antiAliasingTransparency = 0.0f;	// cas très rare car étoile minuscule (point)
                }
                else
                {
                    // Approximation sphere is bigger than voxel
                    //
                    // Evaluate the 1D ratio of distance of intersection
                    antiAliasingTransparency = 1 - ( sphereRadius + Rv - D ) / ( 2 * Rv );
                    // Clamp value
                    //__saturatef( antiAliasingTransparency );
                    clamp( antiAliasingTransparency, 0.0f, 1.0f );
                }


                //*********************************************************************/

                if( cShaderBlurSphere )
                {
                    if(D <= sphereRadius + Rv)
                    {
                        objectTransparency = __saturatef(D / sphereRadius);
                    }
                }


                if ( !cShaderLightSourceType )
                {
                    color *= ( 1 - objectTransparency );
                }
                else
                {
                    objectTransparency = 1.0;
                }

                if( cShading )
                {
                    float3 ambientTerm = .5f * make_float3( cShaderFogColor.x, cShaderFogColor.y, cShaderFogColor.z);
                    float3 finalColor = color * ambientTerm;


                    float3 p = _renderViewContext + lambda * pRayDir;
                    float3 normalVec = normalize( p - (spherePosition+make_float3(brickData.x, brickData.y, brickData.z)) );


                    float3 lightVec = normalize( cLightPosition - p );

                    float lambertTerm = dot( normalVec, lightVec );

                    if ( lambertTerm > 0.0f )
                    {

                        float3 diffuseTerm = make_float3(.6f);
                        // diffuse
                        finalColor += color * diffuseTerm * lambertTerm;

                        float3 halfVec = normalize( lightVec + normalVec );
                        float specular = __powf( max( dot( normalVec, halfVec ), 0.0f ), 64.0f );

                        float3 specularTerm = make_float3(.3f);

                        finalColor += color * make_float3(specular) * specularTerm;
                    }
                    color = finalColor;
                }

                // We want to represent the fog
                if(cShaderFog)
                {
                    float3 fogColor = make_float3(cShaderFogColor.x, cShaderFogColor.y, cShaderFogColor.z);

                    // distance beetween the eye to the sphere
                    float fogDepth = sqrtf(eyeToSphere.x*eyeToSphere.x + eyeToSphere.y*eyeToSphere.y + eyeToSphere.z*eyeToSphere.z);

                    float localFogDepth = fogDepth - _accFogDepth;

                     // Independent order + nb cn be negative
                    _accFogDepth += localFogDepth;
                    float fogTransparency = expf(-cShaderFogDensity * localFogDepth);

                    _accColor += _accTransparency * ( ( 1 - fogTransparency) * fogColor + fogTransparency * color * ( 1 - antiAliasingTransparency ) );
                    _accTransparency *= fogTransparency * ( 1 - ( 1 - antiAliasingTransparency) * ( 1 - objectTransparency ) );
                }
                else{
                    _accColor += _accTransparency * ( color * ( 1 - antiAliasingTransparency ) );
                    _accTransparency *= ( 1 - ( 1 - antiAliasingTransparency) * ( 1 - objectTransparency ) );
                }
            }
        }
		
		// Stop criterion
        if ( 1 - _accTransparency >= cOpacityThreshold )	// stopCriterionImpl() method
        {
           // __saturatef( _accOpacity );
           // _finalColor = make_float4(_accColor.x, _accColor.y, _accColor.z, _accOpacity);
			return;
		}
    }
}
