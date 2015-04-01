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
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
BVHTrianglesManager< TDataTypeList, DataPageSize >::BBoxInfo
::BBoxInfo( uint oid, AABB& bbox/*, float2 minMaxSurf*/ )
:	objectID( oid )
{
	float3 centerPos = bbox.pMin + ( bbox.size() / make_float3( 2.0f ) );
	// Have to ensure that pos are in [0.0, 1.0]
#if 0
	uint3 posHash3D = make_uint3( centerPos * make_float3( (float)( ( 1 << 11 ) - 1 ) ) );  //10
	posHash = interleaveBits( posHash3D );
#else
	uint usefulBits = ( sizeof( PosHasType ) * 8 ) / 3 + 1;

	uint3 posHash3D = make_uint3( centerPos * make_float3( (float)( ( 1 << usefulBits ) - 1 ) ) );  // 10

	uint3 posHashPart[ 3 ];
	posHashPart[ 0 ] = make_uint3( posHash3D.x & 0x000003FF, posHash3D.y & 0x000003FF, posHash3D.z & 0x000003FF);
	posHashPart[ 1 ] = make_uint3( (posHash3D.x & 0x000FFC00)>>10, (posHash3D.y & 0x000FFC00)>>10, (posHash3D.z & 0x000FFC00)>>10);
	posHashPart[ 2 ] = make_uint3( (posHash3D.x & 0x3FF00000)>>20, (posHash3D.y & 0x3FF00000)>>20, (posHash3D.z & 0x3FF00000)>>20);

	uint posHashUI[ 3 ];
	posHashUI[ 0 ] = interleaveBits( posHashPart[ 0 ] );
	posHashUI[ 1 ] = interleaveBits( posHashPart[ 1 ] );
	posHashUI[ 2 ] = interleaveBits( posHashPart[ 2 ] );

	posHash = (PosHasType)posHashUI[ 0 ] | ( ( (PosHasType)posHashUI[ 1 ] ) << 30 ) ;// | ( ((PosHasType)posHashUI[2])<<60 );
#endif
	// May be removed
	/*float normalizedsurface= (bbox.getSurface() - minMaxSurf.x) / (minMaxSurf.y-minMaxSurf.x);
	surfaceHash=(uchar)normalizedsurface*8.0f;*/
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
BVHTrianglesManager< TDataTypeList, DataPageSize >::BBoxInfo
::BBoxInfo( uint oid, float3 v0, float3 v1, float3 v2 )
:	objectID( oid )
{
	float3 centerPos = ( v0 + v1 + v2 ) / make_float3( 3.0f );

	// Have to ensure that pos are in [0.0, 1.0]

	centerPos = min( make_float3( 1.0f ), max( centerPos, make_float3( 0.0f ) ) );

	uint usefulBits = ( sizeof( PosHasType ) * 8 ) / 3 + 1;

	float maxAxisValueF = (float)( ( 1U << usefulBits ) - 1U );
	float3 posHash3DF = centerPos * make_float3( maxAxisValueF );
	posHash3DF = min( make_float3( maxAxisValueF ), max( posHash3DF, make_float3( 0.0f ) ) ); //Clamp

	uint3 posHash3D = make_uint3( posHash3DF );  //10

	uint3 posHashPart[ 3 ];
	posHashPart[ 0 ] = make_uint3( posHash3D.x & 0x000003FF, posHash3D.y & 0x000003FF, posHash3D.z & 0x000003FF);
	posHashPart[ 1 ] = make_uint3( (posHash3D.x & 0x000FFC00)>>10, (posHash3D.y & 0x000FFC00)>>10, (posHash3D.z & 0x000FFC00)>>10);
	posHashPart[ 2 ] = make_uint3( (posHash3D.x & 0x3FF00000)>>20, (posHash3D.y & 0x3FF00000)>>20, (posHash3D.z & 0x3FF00000)>>20);

	uint posHashUI[ 3 ];
	posHashUI[ 0 ] = interleaveBits( posHashPart[ 0 ] );
	posHashUI[ 1 ] = interleaveBits( posHashPart[ 1 ] );
	posHashUI[ 2 ] = interleaveBits( posHashPart[ 2 ] );

	posHash = (PosHasType)posHashUI[ 0 ]; //|  ( ((PosHasType)posHashUI[1])<<30 )  | ( ((PosHasType)posHashUI[2])<<60 );
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
bool BVHTrianglesManager< TDataTypeList, DataPageSize >::BBoxInfo
::operator<( const BBoxInfo& b ) const
{
#if 1
	return posHash < b.posHash;
#else
	if(posHash[2]<b.posHash[2]){
		return true;
	}else if(posHash[2]>b.posHash[2]){
		return false;
	}else {//[2]==[2]
		if(posHash[1]<b.posHash[1]){
			return true;
		}else if(posHash[1]>b.posHash[1]){
			return false;
		}else {//[1]==[1]
			if(posHash[0]<b.posHash[0]){
				return true;
			}else {
				return false;
			}
		}
	}

#endif
}

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
BVHTrianglesManager< TDataTypeList, DataPageSize >
::BVHTrianglesManager()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
BVHTrianglesManager< TDataTypeList, DataPageSize >
::~BVHTrianglesManager()
{
}

/******************************************************************************
 * Helper function to stringify a value
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline std::string BVHTrianglesManager< TDataTypeList, DataPageSize >
::stringify( int x )
{
	std::ostringstream o;
	o << x;

	return o.str();
}

/******************************************************************************
 * Iterate through triangles and split them if required (depending on a size criteria)
 *
 * @param criticalEdgeLength max "triangle edge length" criteria beyond which a split must occur
 *
 * @return flag to tell wheter or not a split has happend
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline bool BVHTrianglesManager< TDataTypeList, DataPageSize >
::splitTrianges( float criticalEdgeLength )
{
	// Flag to tell wheter or not a split has happend
	bool splitAppend = false;

	// Iterate through triangles
	uint triangleListSize = meshTriangleList.size();
	for ( uint i = 0; i < triangleListSize; ++i )
	{
		// Retrieve triangle's positions
		float3 vertPos[ 3 ];
		vertPos[ 0 ] = meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 0 ] ];
		vertPos[ 1 ] = meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 1 ] ];
		vertPos[ 2 ] = meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 2 ] ];

		// Generate triangle's edges
		float3 edges[ 3 ];
		edges[ 0 ] = vertPos[ 1 ] - vertPos[ 0 ];
		edges[ 1 ] = vertPos[ 2 ] - vertPos[ 1 ];
		edges[ 2 ] = vertPos[ 0 ] - vertPos[ 2 ];

		// Check size criteria to decide wheter or not to split triangle
		float maxEdgeLength = maxcc( length( edges[ 0 ] ), maxcc( length( edges[ 1 ] ), length( edges[ 2 ] ) ) );
		if ( maxEdgeLength >= criticalEdgeLength )
		{
			// Update flag to tell wheter that a split has happend
			splitAppend = true;

			// Iterate through triangle edges and retrieve the index of the longest one
			uint edgeNum = 0;
			for ( uint e = 0; e < 3; ++e )
			{
				if ( length( edges[ e ] ) == maxEdgeLength )
				{
					edgeNum = e;
				}
			}

			uint newVertIndex = meshVertexPositionList.size();
			//std::cout << meshVertexPositionList.size() << " " << meshVertexColorList.size() << " !\n";

			// Retrieve triangle's colors
			float4 vertColor[ 3 ];
			vertColor[ 0 ] = meshVertexColorList[ meshTriangleList[ i ].vertexID[ 0 ] ];
			vertColor[ 1 ] = meshVertexColorList[ meshTriangleList[ i ].vertexID[ 1 ] ];
			vertColor[ 2 ] = meshVertexColorList[ meshTriangleList[ i ].vertexID[ 2 ] ];

			// Split process
			//
			// Split triangle in two, compute new position and its attributes (interpolated color, etc...)

			// Split triangle : modify triangle and add new one
			Triangle newTriangle0;
			Triangle newTriangle1;
			for ( uint v = 0; v < 3; ++v )
			{
				if ( v == edgeNum )
					newTriangle0.vertexID[ v ] = newVertIndex;
				else
					newTriangle0.vertexID[ v ] = meshTriangleList[ i ].vertexID[ v ];

				if ( v == ( edgeNum + 1 ) % 3 )
					newTriangle1.vertexID[ v ] = newVertIndex;
				else
					newTriangle1.vertexID[ v ] = meshTriangleList[ i ].vertexID[ v ];
			}
			meshTriangleList[ i ] = newTriangle0;		// Replaced one
			meshTriangleList.push_back( newTriangle1 );	// Push back new

			// Compute and add new position
			float3 newVertPos = vertPos[ edgeNum ] + ( edges[ edgeNum ] / make_float3( 2.0f ) );
			meshVertexPositionList.push_back( newVertPos );

			// Compute and add new color
			float4 newVertColor = ( vertColor[ edgeNum ] + vertColor[ ( edgeNum + 1 ) % 3 ] ) / make_float4( 2.0f );
			//float4 newVertColor = make_float4( 1.0f );
			meshVertexColorList.push_back( newVertColor );

			// break;
		}
	}

	return splitAppend;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline std::string BVHTrianglesManager< TDataTypeList, DataPageSize >
::getBaseFileName( const std::string& fileName )
{
	std::string baseFileName;

	int ppos = fileName.rfind( "." );
	int slashpos = fileName.rfind( "/" );
	if ( ppos > slashpos )
		baseFileName = fileName.substr( 0, slashpos );
	else
		baseFileName = fileName;

	return baseFileName;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::loadPowerPlant( const std::string& baseFileName )
{
	loadPowerPlantDirectoryStructure( baseFileName );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::loadMesh( const std::string& meshFileName )
{
	this->addMeshFile( meshFileName );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::saveRawMesh( const std::string& fileName )
{
	std::string baseFileName = getBaseFileName( fileName );

	std::string serialFileName;

	// saving
	serialFileName = baseFileName + "/meshVertexPosition.dat";
	writeStdVector( serialFileName, meshVertexPositionList );

	serialFileName = baseFileName + "/meshVertexColor.dat";
	writeStdVector( serialFileName, meshVertexColorList );

	serialFileName = baseFileName + "/meshTriangle.dat";
	writeStdVector( serialFileName, meshTriangleList );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::loadRawMesh( const std::string& fileName )
{
	std::string baseFileName = getBaseFileName( fileName );

	std::string serialFileName;
	//Loading
	serialFileName = baseFileName + "/meshVertexPosition.dat";
	readStdVector( serialFileName, meshVertexPositionList );

	serialFileName = baseFileName + "/meshVertexColor.dat";
	readStdVector( serialFileName, meshVertexColorList );

	serialFileName = baseFileName + "/meshTriangle.dat";
	readStdVector( serialFileName, meshTriangleList );
}

/******************************************************************************
 * Generate buffers
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::generateBuffers( uint arrayFlag )
{
	// Correct bounds
	float3 minPos = make_float3( 1000000.0f );
	float3 maxPos = make_float3( -1000000.0f );

	// Compute BBox
	for ( uint v = 0; v < meshVertexPositionList.size(); ++v )
	{
		float3 pos = meshVertexPositionList[v];

		minPos = min( minPos, pos );
		maxPos = max( maxPos, pos );
	}

	std::cout << minPos << " " << maxPos << "\n";
	float3 bboxSize = maxPos - minPos;
	float maxBBoxSize = maxcc( bboxSize.x, maxcc( bboxSize.y, bboxSize.z ) );

	for ( uint v = 0; v < meshVertexPositionList.size(); ++v )
	{
		float3 pos = meshVertexPositionList[v];

		pos  = ( pos - minPos ) / make_float3( maxBBoxSize );

		meshVertexPositionList[ v ] = pos;
	}

#if 1
	// Split large triangles
	bool splitAppend = false;
	do
	{
		splitAppend = this->splitTrianges( 1.0f / 1.0f ); //16 //64: Seems there is still big triangles...
		std::cout << "Split ";
	}
	while ( splitAppend );		// WARNING !! => infinity !!!
	std::cout << "\n";
#endif
	std::vector< AABB > trianglesAABBList;

	//float2 minMaxSurf=make_float2(1000000.0f, -1000000.0f);

	// Compute triangles AABB
	for ( uint i = 0; i < meshTriangleList.size(); ++i )
	{
		Triangle triangle = meshTriangleList[ i ];

		float3 minpos = min( meshVertexPositionList[ triangle.vertexID[ 0 ] ],
			min( meshVertexPositionList[ triangle.vertexID[ 1 ] ],
			meshVertexPositionList[ triangle.vertexID[ 2 ] ] ) );

		float3 maxpos = max( meshVertexPositionList[ triangle.vertexID[ 0 ] ],
			max( meshVertexPositionList[ triangle.vertexID[ 1 ] ],
			meshVertexPositionList[ triangle.vertexID[ 2 ] ] ) );

		AABB triangleAABB;
		triangleAABB.init( minpos, maxpos );

		//Compute volume minmax
		/*float surface=triangleAABB.getSurface();
		if(surface>minMaxSurf.y)
		minMaxSurf.y=surface;
		else if(surface<minMaxSurf.x)
		minMaxSurf.x=surface;*/

		trianglesAABBList.push_back( triangleAABB );
	}

	// Compute triangles BBox info

	std::vector< BBoxInfo > bboxInfoList;

	// Compute bbox info
	for ( uint i = 0; i < meshTriangleList.size(); ++i )
	{
		//BBoxInfo bboxInfo(i, trianglesAABBList[i]/*, minMaxVol*/);
		BBoxInfo bboxInfo( i,
			meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 0 ] ],
			meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 1 ] ],
			meshVertexPositionList[ meshTriangleList[ i ].vertexID[ 2 ] ]);

		bboxInfoList.push_back( bboxInfo );
	}

	// Sort
	std::sort( bboxInfoList.begin(), bboxInfoList.end() );

	// Create triangle pages with associated nodes
	std::vector< VolTreeBVHNode > bvhNodesList;

	std::vector< uint > trianglePagesTemp;	// Building location for triangle pages

	uint numTrianglesPerPage = DataPageSize / 3;

	/*Triangle	nullTriangle;
	nullTriangle.vertexID[0]=0;
	nullTriangle.vertexID[1]=0;
	nullTriangle.vertexID[2]=0;*/

	float3 nodeStartPos = make_float3( 1000000.0f );
	float3 nodeEndPos = make_float3( -1000000.0f );

	uint curPageNumber = 0;
	uint curOffsetInPage = 0;

	std::cout << "meshTriangleList.size " << meshTriangleList.size() << "\n";
	std::cout << "bboxInfoList.size " << bboxInfoList.size() << "\n";

	float3 lastTrianglePos = trianglesAABBList[ bboxInfoList[ 0 ].objectID ].center();
	const float criticalDist = 1.0f / 32.0f; //512.0f; //256.0f

	bool pageStart = true;

	uint i = 0;
	do {	//Iterate among triangles info
		AABB triangleAABB;
		triangleAABB = trianglesAABBList[ bboxInfoList[ i ].objectID ];

		float distToLastTriangle;
		if ( !pageStart )
			distToLastTriangle = length( triangleAABB.center() - lastTrianglePos );
		else
			distToLastTriangle = 0.0f;

		lastTrianglePos = triangleAABB.center();
		pageStart = false;

		// fill current page
		if ( true && distToLastTriangle > criticalDist )
		{
			for ( uint e = (curOffsetInPage); e < numTrianglesPerPage; ++e )
			{
				trianglePagesTemp.push_back( 0 );
				trianglePagesTemp.push_back( 0 );
				trianglePagesTemp.push_back( 0 );
			}
			curOffsetInPage = (numTrianglesPerPage);
		}
		else
		{
			nodeStartPos = min( nodeStartPos, triangleAABB.pMin );
			nodeEndPos = max( nodeEndPos, triangleAABB.pMax );

			Triangle tri = meshTriangleList[ bboxInfoList[ i ].objectID ];
			trianglePagesTemp.push_back( tri.vertexID[ 0 ] );
			trianglePagesTemp.push_back( tri.vertexID[ 1 ] );
			trianglePagesTemp.push_back( tri.vertexID[ 2 ] );	

			curOffsetInPage++;

			i++;
		}

		// last triangle -> fill
		if ( i == ( bboxInfoList.size() ) )
		{
			for ( uint e = (curOffsetInPage) * 3; e < DataPageSize; ++e )
			{
				trianglePagesTemp.push_back( 0 );
			}
			curOffsetInPage = (numTrianglesPerPage);
		}

		//Page + node creation
		if ( curOffsetInPage == ( numTrianglesPerPage ) )
		{
			VolTreeBVHNode bvhNode;

			const float offsetBug = 1.0f / 16777216.0f;
			if ( nodeStartPos.x == nodeEndPos.x )
			{
				nodeStartPos.x -= offsetBug;
				nodeEndPos.x += offsetBug;
			}
			else if ( nodeStartPos.y == nodeEndPos.y )
			{
				nodeStartPos.y -= offsetBug;
				nodeEndPos.y += offsetBug;
			}else if ( nodeStartPos.z == nodeEndPos.z )
			{
				nodeStartPos.z -= offsetBug;
				nodeEndPos.z += offsetBug;
			}

			bvhNode.userNode.bbox.pMin = nodeStartPos;
			bvhNode.userNode.bbox.pMax = nodeEndPos;
			bvhNode.userNode.setDataIdx( curPageNumber /* *DataPageSize*/);
			//std::cout << curPageNumber<<" ";

			bvhNode.userNode.setCPULink();
			bvhNodesList.push_back( bvhNode );

			nodeStartPos = make_float3( 1000000.0f );
			nodeEndPos = make_float3( -1000000.0f );

			pageStart = true;

			// Fill the page fully
			for ( uint e = (curOffsetInPage) * 3; e < DataPageSize; ++e )
			{
				trianglePagesTemp.push_back( 0 );
			}

			curPageNumber++;
			curOffsetInPage = 0;
		}
	}
	while ( i < bboxInfoList.size() );

	std::cout << curPageNumber << "\n\n";

	/*if(bvhNodesList.size()%2==1){
	VolTreeBVHNode bvhNode;

	bvhNode.userNode.bbox.pMin=make_float3(1000000.0f);
	bvhNode.userNode.bbox.pMax=make_float3(-1000000.0f);
	bvhNode.userNode.setDataIdx(0);

	bvhNode.userNode.setCPULink();
	bvhNodesList.push_back(bvhNode);
	}*/

	std::cout<<"trianglePagesTemp.size "<<trianglePagesTemp.size()/3<<"\n";
	std::cout<<"bvhNodesList.size "<<bvhNodesList.size()*DataPageSize/3<<"\n";

	////Build final triangle data pool////
	//_dataBuffer= new DataBufferType(make_uint3(trianglePagesTemp.size()*3), 2);	//Allocate mapped buffer

	//Compute nodes BBox info
	bboxInfoList.clear();

	// Compute bbox info
	for ( uint i = 0; i < bvhNodesList.size(); ++i )
	{
		BBoxInfo bboxInfo( i, bvhNodesList[ i ].userNode.bbox );

		bboxInfoList.push_back( bboxInfo );
	}

	// Sort
	std::sort( bboxInfoList.begin(), bboxInfoList.end() );

	// Build nodes buffer
	std::vector< VolTreeBVHNode > bvhNodesBufferBuildingList;

	PosHasType biggestHash = bboxInfoList.back().posHash;

	int bitPos = sizeof( PosHasType ) * 8 - 1;
	PosHasType bitTest = biggestHash & ( PosHasType( 1 ) << bitPos );
	while ( !bitTest )
	{
		bitPos--;
		bitTest = biggestHash & ( PosHasType( 1 ) << bitPos );
	}

	bvhNodesBufferBuildingList.reserve( 2 * bvhNodesList.size() );
	//std::cout << "Reserve: " << bvhNodesBufferBuildingList.capacity() << "\n";

	uint2 startInterval = make_uint2( 0, bboxInfoList.size() );
	uint nodeBufferOffset = 2;	//Start at 2
	// TODO: put a link to the first tile at 2
	bvhNodesBufferBuildingList.push_back( VolTreeBVHNode() );
	bvhNodesBufferBuildingList.push_back( VolTreeBVHNode() );

	numDataNodesCounter = 0;
	fillNodesBuffer( bvhNodesList, bboxInfoList,
		bitPos, startInterval, nodeBufferOffset, bvhNodesBufferBuildingList );

	std::cout<<"numDataNodesCounter "<<numDataNodesCounter<<"\n";

	// Fill final nodes buffer
	_nodesBuffer = new GvCore::Array3D< VolTreeBVHNode >( make_uint3( bvhNodesBufferBuildingList.size(), 1, 1 ), arrayFlag );	// Mapped memory
	for ( uint i = 0; i < bvhNodesBufferBuildingList.size(); ++i )
	{
		VolTreeBVHNode node = bvhNodesBufferBuildingList[ i ];
		//node.userNode.setGPULink();
		_nodesBuffer->get( i ) = node;
	}
	std::cout << "NodesBufferSize: " << bvhNodesBufferBuildingList.size() << "\n";

	// Escape index computation
	recursiveAddEscapeIdx( 2, 3 );
	recursiveAddEscapeIdx( 3, 0 );

	// Fill final vertex buffer
	uint vertexBufferSize = trianglePagesTemp.size();
	_dataBuffer = new GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >( make_uint3( vertexBufferSize, 1, 1 ), arrayFlag );

	for ( uint i = 0; i < trianglePagesTemp.size(); ++i )
	{
		float3 vertPos = meshVertexPositionList[ trianglePagesTemp[ i ] ];
		float4 vertPosStore;
		vertPosStore.x = vertPos.x; vertPosStore.y = vertPos.y; vertPosStore.z = vertPos.z; vertPosStore.w = 1.0f;

		_dataBuffer->getChannel( Loki::Int2Type< 0 >() )->get( i ) = vertPosStore;

#if BVH_USE_COLOR
		float4 vertColor = meshVertexColorList[ trianglePagesTemp[ i ] ];
		uchar4 vertColorStore;
		vertColorStore.x = uchar( vertColor.x * 255.0f );
		vertColorStore.y = uchar( vertColor.y * 255.0f );
		vertColorStore.z = uchar( vertColor.z * 255.0f );
		vertColorStore.w = uchar( vertColor.w * 255.0f );

		_dataBuffer->getChannel( Loki::Int2Type< 1 >() )->get( i ) = vertColorStore;
#endif
	}
	std::cout << "VertexBufferSize: " << vertexBufferSize << "\n";
}

// TODO: case where data is linked
/******************************************************************************
 * Fill nodes buffer
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline VolTreeBVHNode BVHTrianglesManager< TDataTypeList, DataPageSize >
::fillNodesBuffer( std::vector< VolTreeBVHNode >& bvhNodesList, std::vector< BBoxInfo >& bboxInfoList,
					int level, uint2 curInterval, uint& nextNodeBufferOffset,
					std::vector< VolTreeBVHNode >& bvhNodesBufferBuildingList )
{
	VolTreeBVHNode	 currentNode;

	//std::cout << "bitPos: " << level << " Interval [" << curInterval.x << ", " << curInterval.y << "]\n";

	if ( curInterval.y - curInterval.x <= 1 /*|| level==0*/ )
	{
		currentNode = bvhNodesList[ bboxInfoList[ curInterval.x ].objectID ];
		//std::cout<<bboxInfoList[curInterval.x].objectID<<"\n";
		numDataNodesCounter++;
		//std::cout<<level<<"\n";

	}
	else
	{	//There will be subnodes
		uint split;
		if ( level >= 0 )
		{
			PosHasType bitMask = ( PosHasType( 1 ) << level );

			split = curInterval.x;
			PosHasType compRes = bboxInfoList[ split ].posHash & bitMask;

			while( !compRes && split < curInterval.y )
			{
				split++;

				/*if(split>=bboxInfoList.size())
				std::cout<<split<<" "<<bboxInfoList.size()<<" "<<curInterval.y<<"\n";*/
				if ( split < bboxInfoList.size() )
					compRes = bboxInfoList[ split ].posHash & bitMask;
			}
		}
		else
		{
			split = curInterval.x + ( curInterval.y - curInterval.x ) / 2;
		}

		uint curNodeBufferOffset = nextNodeBufferOffset;
		// Reserve only if a node is needed
		if ( split > curInterval.x && split < curInterval.y )
		{
			// Reserve space
			bvhNodesBufferBuildingList.push_back( VolTreeBVHNode() );
			bvhNodesBufferBuildingList.push_back( VolTreeBVHNode() );

			// Reserve two slots
			nextNodeBufferOffset += 2;

			bvhNodesBufferBuildingList[ curNodeBufferOffset ]
			= fillNodesBuffer( bvhNodesList, bboxInfoList, level - 1,
				make_uint2( curInterval.x, split ), nextNodeBufferOffset,
				bvhNodesBufferBuildingList );

			bvhNodesBufferBuildingList[ curNodeBufferOffset + 1 ]
			= fillNodesBuffer( bvhNodesList, bboxInfoList, level - 1,
				make_uint2( split, curInterval.y ), nextNodeBufferOffset,
				bvhNodesBufferBuildingList );

			AABB bboxres;
			AABB bbox[ 2 ];
			bbox[ 0 ] = bvhNodesBufferBuildingList[ curNodeBufferOffset ].userNode.bbox;
			bbox[ 1 ] = bvhNodesBufferBuildingList[ curNodeBufferOffset + 1 ].userNode.bbox;

			bboxres.pMin = min( bbox[ 0 ].pMin, bbox[ 1 ].pMin );
			bboxres.pMax = max( bbox[ 0 ].pMax, bbox[ 1 ].pMax );

			currentNode.userNode.setSubNodeIdx( curNodeBufferOffset );
			currentNode.userNode.bbox = bboxres;
		}
		else if ( split > curInterval.x )
		{
			currentNode = fillNodesBuffer( bvhNodesList, bboxInfoList, level - 1,
				make_uint2( curInterval.x, split ), nextNodeBufferOffset,
				bvhNodesBufferBuildingList );

		}
		else
		{
			currentNode = fillNodesBuffer( bvhNodesList, bboxInfoList, level - 1,
				make_uint2( split, curInterval.y ), nextNodeBufferOffset,
				bvhNodesBufferBuildingList );
		}
	}

	currentNode.userNode.setCPULink();

	return currentNode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::recursiveAddEscapeIdx( uint nodeAddress, uint escapeTo )
{
	_nodesBuffer->get( nodeAddress ).userNode.escapeIdx = escapeTo;

	VolTreeBVHNodeUser node = _nodesBuffer->get( nodeAddress ).userNode;
	if ( node.hasSubNodes() )
	{
		uint childaddress = node.getSubNodeIdx();

		recursiveAddEscapeIdx( childaddress, childaddress + 1 );
		recursiveAddEscapeIdx( childaddress + 1, escapeTo );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::loadPowerPlantDirectoryStructure( const std::string& baseFileName )
{
	struct stat buf;

	int dir0Exists;
	int dir0Num = 1;
	do
	{
		std::string dir0Name = baseFileName + "/ppsection" + stringify( dir0Num );
		dir0Exists = !stat( dir0Name.c_str(), &buf );

		if ( dir0Exists )
		{
			int dir1Exists;
			int dir1Num = 0;
			do
			{
				std::string dir1Name = dir0Name + "/part_";
				dir1Name.push_back( 'a' + dir1Num );
				dir1Exists = !stat( dir1Name.c_str(), &buf );

				if ( dir1Exists )
				{
					int fileExists;
					int fileNum=0;
					do
					{
						std::string meshFileName = dir1Name + "/g" + stringify( fileNum ) + ".ply";
						fileExists = !stat( meshFileName.c_str(), &buf );

						if ( fileExists )
						{
							std::cout << "Loading :" << meshFileName << "\n";
							this->addMeshFile( meshFileName );
						}

						fileNum++;
					}
					while( fileExists );

				}

				dir1Num++;
			}
			while( dir1Exists );

		}

		dir0Num++;
	}
	while( dir0Exists );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::addMeshFile( const std::string& meshFileName )
{
	const aiScene* pScene;

	// Create an instance of the Importer class
	Assimp::Importer importer;

	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll
	// propably to request more postprocessing than we do in this example.
	/*const aiScene* scene = importer.ReadFile( meshFileName,
	aiProcess_CalcTangentSpace	   |
	aiProcess_Triangulate			|
	aiProcess_JoinIdenticalVertices  |
	aiProcess_SortByPType);*/
	pScene = importer.ReadFile( meshFileName, aiProcessPreset_TargetRealtime_Fast );

	// If the import failed, report it
	if( !pScene)	{
		std::cout<< importer.GetErrorString();
		return ;
	}

	float xmin = +FLT_MAX;
	float ymin = +FLT_MAX;
	float zmin = +FLT_MAX;
	float xmax = -FLT_MAX;
	float ymax = -FLT_MAX;
	float zmax = -FLT_MAX;

	// determine the bounding box of the mesh
	for (unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = pScene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices ; ++vertexIndex)
		{
			xmin = std::min<float>(pMesh->mVertices[vertexIndex].x, xmin);
			ymin = std::min<float>(pMesh->mVertices[vertexIndex].y, ymin);
			zmin = std::min<float>(pMesh->mVertices[vertexIndex].z, zmin);
			xmax = std::max<float>(pMesh->mVertices[vertexIndex].x, xmax);
			ymax = std::max<float>(pMesh->mVertices[vertexIndex].y, ymax);
			zmax = std::max<float>(pMesh->mVertices[vertexIndex].z, zmax);
		}
	}

	// scale the mesh according to the bounding box
	float scale = 0.95f / std::max<float>(std::max<float>(xmax - xmin, ymax - ymin), zmax - zmin);

	for (unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = pScene->mMeshes[meshIndex];

		// this is used as an offset for faces
		size_t faceOffset = meshVertexPositionList.size();

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices ; ++vertexIndex)
		{
			float3 position;

			position.x = (pMesh->mVertices[vertexIndex].x - 0.5f * (xmax + xmin)) * scale + 0.5f;
			position.y = (pMesh->mVertices[vertexIndex].y - 0.5f * (ymax + ymin)) * scale + 0.5f;
			position.z = (pMesh->mVertices[vertexIndex].z - 0.5f * (zmax + zmin)) * scale + 0.5f;
			meshVertexPositionList.push_back(position);

			float4 color = make_float4(0.9f, 0.9f, 0.9f, 1.0f);

			if (pMesh->HasVertexColors(0))
				color = make_float4(pMesh->mColors[0][vertexIndex].r, pMesh->mColors[0][vertexIndex].g, pMesh->mColors[0][vertexIndex].b, pMesh->mColors[0][vertexIndex].a);

			meshVertexColorList.push_back(color);
		}

		for (unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex)
		{
			const struct aiFace *pFace = &pMesh->mFaces[faceIndex];

			// we don't support polygons
			assert(pFace->mNumIndices == 3);

			Triangle triangle;

			triangle.vertexID[0] = pFace->mIndices[0] + faceOffset;
			triangle.vertexID[1] = pFace->mIndices[1] + faceOffset;
			triangle.vertexID[2] = pFace->mIndices[2] + faceOffset;

			meshTriangleList.push_back(triangle);
		}
	}

	// Now we can access the file's contents.
	//DoTheSceneProcessing( scene);

	//std::cout<<aiscene->mRootNode->mNumMeshes<<" !\n";

	//	  uint previousNumVertex=meshVertexPositionList.size();

	//	  aiMesh* aimesh  = aiscene->mMeshes[0];

	//bool hasColor=aimesh->GetNumColorChannels();

	//	  for(uint v=0; v < aimesh->mNumVertices; ++v){
	//		  float3 pos=make_float3(aimesh->mVertices[v].x, aimesh->mVertices[v].y, aimesh->mVertices[v].z);

	//		  meshVertexPositionList.push_back( pos );

	//		  float4 color;
	//	if(hasColor)
	//		color=make_float4(aimesh->mColors[0][v].r, aimesh->mColors[0][v].g, aimesh->mColors[0][v].b, aimesh->mColors[0][v].a);
	//	else
	//		color=make_float4(0.9f, 0.9f, 0.9f, 1.0f);
	//		  meshVertexColorList.push_back( color );


	//	  }
	//	  //std::cout<<minPos<<" -> "<<maxPos<<"\n";

	//	  for(uint t=0; t < aimesh->mNumFaces; ++t){
	//		  Triangle triangle;
	//		  triangle.vertexID[0]=aimesh->mFaces[t].mIndices[0] +previousNumVertex;
	//		  triangle.vertexID[1]=aimesh->mFaces[t].mIndices[1] +previousNumVertex;
	//		  triangle.vertexID[2]=aimesh->mFaces[t].mIndices[2] +previousNumVertex;

	//		  meshTriangleList.push_back( triangle );
	//	  }
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::renderGL()
{
	//glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

#if 1
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (float*)&(meshVertexPositionList[0]));

	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_FLOAT, 0, (float*)&(meshVertexColorList[0]));

	glDrawElements(GL_TRIANGLES, meshTriangleList.size()*3, GL_UNSIGNED_INT, (uint*)&(meshTriangleList[0]) );

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
#else
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, (float*) _dataBuffer->getChannel(Loki::Int2Type<0>())->getPointer(0)  );

	glDrawArrays(GL_TRIANGLES, 0, _dataBuffer->getChannel(Loki::Int2Type<0>())->getNumElements()/100 );

	glDisableClientState(GL_VERTEX_ARRAY);
#endif
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::renderDebugGL()
{
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	
#if 1
	recursiveRenderDebugGL(2);
#else
	glBegin(GL_QUADS);
	for(uint i=0; i<this->_nodesBuffer->getSize().x; ++i){
		VolTreeBVHNode node=this->_nodesBuffer->get(i);
		if(node.userNode.isDataType())
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
		else
			glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
		drawCubeQuadPrimStarted(node.userNode.bbox.position, node.userNode.bbox.position+node.userNode.bbox.size);
	}

	glEnd();
#endif
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::recursiveRenderDebugGL( uint curNodeIdx )
{
	VolTreeBVHNode node;

	node=this->_nodesBuffer->get(curNodeIdx);
	//std::cout<<"r: "<<curNodeIdx<<" n: "<<node.userNode.subNodesDataIdx<<"\n";

	if(node.userNode.isLinkActive()){
		if(node.userNode.isDataType()){
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
			drawCube(node.userNode.bbox.pMin, node.userNode.bbox.pMax);
		}else{
			recursiveRenderDebugGL(node.userNode.getSubNodeIdx());

			glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
			drawCube(node.userNode.bbox.pMin, node.userNode.bbox.pMax);

		}
	}

	node=this->_nodesBuffer->get(curNodeIdx+1);
	if(node.userNode.isLinkActive()){
		if(node.userNode.isDataType()){
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
			drawCube(node.userNode.bbox.pMin, node.userNode.bbox.pMax);
		}else{
			recursiveRenderDebugGL(node.userNode.getSubNodeIdx());

			glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
			drawCube(node.userNode.bbox.pMin, node.userNode.bbox.pMax);
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::displayTriangles( uint index )
{
	//std::cout<<"["<<index<<"]: ";
	for ( uint i = 0; i < ( BVH_DATA_PAGE_SIZE / 3 ) * 3; ++i )
	{
		//std::cout<<i<<" ";
		float4 vpos = _dataBuffer->getChannel( Loki::Int2Type< 0 >() )->get( index * BVH_DATA_PAGE_SIZE + i );

		uchar4 vcol = _dataBuffer->getChannel( Loki::Int2Type< 1 >() )->get( index * BVH_DATA_PAGE_SIZE + i );
		glColor4f( vcol.x / 255.0f, vcol.y / 255.0f, vcol.z / 255.0f, vcol.w / 255.0f );
		glVertex3f( vpos.x, vpos.y, vpos.z );
	}
	//std::cout<<"\n";
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::renderFullGL()
{
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

	//
#if 0
	glBegin(GL_TRIANGLES);
	recursiveRenderFullGL(2);
	glEnd();
#else
	for(uint i=2; i<this->_nodesBuffer->getResolution().x; ++i){
		VolTreeBVHNode node;

		node=this->_nodesBuffer->get(i);
		if(node.userNode.isDataType()){

			glColor4f(0.0f, 0.8f, 0.0f, 1.0f);
#if 1
			glBegin(GL_TRIANGLES);
			displayTriangles(node.userNode.getDataIdx());
			glEnd();
#else
			if(node.userNode.bbox.pMin.x<0.0f || node.userNode.bbox.pMin.x>1.0f)
				std::cout<<node.userNode.bbox.pMin<<" "<<node.userNode.bbox.pMax<<" ("<<node.userNode.isDataType()<<")\n";
			drawCube(node.userNode.bbox.pMin, node.userNode.bbox.pMax);
#endif

		}
	}

	std::cout<<"\n\n";
#endif
	//
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline void BVHTrianglesManager< TDataTypeList, DataPageSize >
::recursiveRenderFullGL( uint curNodeIdx )
{
	VolTreeBVHNode node;

	node=this->_nodesBuffer->get(curNodeIdx);
	//std::cout<<"r: "<<curNodeIdx<<" n: "<<node.userNode.subNodesDataIdx<<"\n";

	if(node.userNode.bbox.pMin.x<0.0f || node.userNode.bbox.pMin.x>1.0f)
		std::cout<<node.userNode.bbox.pMin<<" "<<node.userNode.bbox.pMax<<" ("<<node.userNode.isDataType()<<")\n";

	if(node.userNode.isLinkActive()){
		if(node.userNode.isDataType()){
			glColor4f(0.0f, 0.8f, 0.0f, 1.0f);
			displayTriangles(node.userNode.getDataIdx());

			//drawCube(node.userNode.bbox.position, node.userNode.bbox.position+node.userNode.bbox.size);
		}else{
			recursiveRenderFullGL(node.userNode.getSubNodeIdx());

			//glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
			//drawCube(node.userNode.bbox.position, node.userNode.bbox.position+node.userNode.bbox.size);

		}
	}

	if(true){
		node=this->_nodesBuffer->get(curNodeIdx+1);
		if(node.userNode.isLinkActive()){
			if(node.userNode.isDataType()){
				glColor4f(0.0f, 0.8f, 0.0f, 1.0f);
				displayTriangles(node.userNode.getDataIdx());

				//drawCube(node.userNode.bbox.position, node.userNode.bbox.position+node.userNode.bbox.size);
			}else{
				recursiveRenderFullGL(node.userNode.getSubNodeIdx());

				//glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
				//drawCube(node.userNode.bbox.position, node.userNode.bbox.position+node.userNode.bbox.size);
			}
		}
	}
}

/******************************************************************************
 * Get the node pool
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline GvCore::Array3D< VolTreeBVHNode >* BVHTrianglesManager< TDataTypeList, DataPageSize >
::getNodesBuffer()
{
	return _nodesBuffer;
}

/******************************************************************************
 * Get the data pool
 ******************************************************************************/
template< class TDataTypeList, uint DataPageSize >
inline BVHTrianglesManager< TDataTypeList, DataPageSize >::DataBufferType* BVHTrianglesManager< TDataTypeList, DataPageSize >
::getDataBuffer()
{
	return _dataBuffer;
}
