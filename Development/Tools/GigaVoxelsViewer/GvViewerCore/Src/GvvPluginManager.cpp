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

#include "GvvPluginManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPluginInterface.h"

// System
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>

// Dynamic library
#ifdef _WIN32
	//#include <windows.h>
#else
	#include <dirent.h>
	#include <dlfcn.h>
#endif

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// // GvViewer
using namespace GvViewerCore;

// STL
//using std::string;
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique instance of the singleton.
 */
GvvPluginManager* GvvPluginManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

#ifdef _WIN32

#else

/******************************************************************************
 * gvPluginfilter()
 *
 * @param pDir
 *
 * @return
 ******************************************************************************/
int gvPluginfilter( const struct dirent* pDirent )
{
    //** GvPlugin's name end with ".gvp"
    int lLength = strlen( pDirent->d_name );
    if ( lLength < 4 )
    {
        return 0;
    }

    const char* lName = pDirent->d_name;
    if ( lName[ --lLength ] == 'p' && lName[ --lLength ] == 'v' && lName[ --lLength ] == 'g' && lName[ --lLength ] == '.' )
    {
        return 1;
    }

    return 0;
}

#endif

/******************************************************************************
 * getInstance()
 *
 * @return
 ******************************************************************************/
GvvPluginManager& GvvPluginManager::get()
{
    if ( msInstance == NULL )
    {
        msInstance = new GvvPluginManager();
    }

    return *msInstance;
}

/******************************************************************************
 * GvPluginManager()
 ******************************************************************************/
GvvPluginManager::GvvPluginManager()
{
}

/******************************************************************************
 * loadPlugins()
 *
 * @param pDir
 ******************************************************************************/
void GvvPluginManager::loadPlugins( const string& pDir )
{
    vector< string > lFilenames;
    getFilenames( pDir, lFilenames );

    //   cout << "Nombre de plugins potentiels trouvés : " << lFilenames.size() << endl;

    vector< string >::const_iterator it;
    for ( it = lFilenames.begin(); it != lFilenames.end(); ++it )
    {
        const string& lFilename = *it;
       // const string& lFullName = pDir + string( "\\" ) + lFilename;
        const string& lFullName = pDir + string( "/" ) + lFilename;
        loadPlugin( lFullName );
    }
}

/******************************************************************************
 * getFilenames()
 *
 * @param pDir
 * @param pFilenames
 ******************************************************************************/
void GvvPluginManager::getFilenames( const string& pDir, vector< string >& pFilenames ) const
{
#ifdef _WIN32

#else

    struct dirent** lNamelist;

    int lNbEntries = scandir( pDir.c_str(), &lNamelist, gvPluginfilter, alphasort );
    if ( lNbEntries < 0 )
    {
      //        cout << "GvPluginManager::getFilenames : Error while using scandir() function." << endl;
    }
    else
    {
        while ( lNbEntries-- )
        {
	  //    printf( "%s\n", lNamelist[ lNbEntries ]->d_name );
            pFilenames.push_back( string( lNamelist[ lNbEntries ]->d_name ) );
           
            free( lNamelist[ lNbEntries ] );
        }
        free( lNamelist );
    }

#endif
}

/******************************************************************************
 * loadPlugin()
 *
 * @param pFilename
 *
 * @return
 ******************************************************************************/
bool GvvPluginManager::loadPlugin( const string& pFilename )
{
	GVV_CREATE_PLUGIN lFunc;			// Function pointer

#ifdef _WIN32

	HINSTANCE lHandle = LoadLibrary( pFilename.c_str() );
	if ( lHandle == NULL )
	{
		return false;
	}
	
	lFunc = (GVV_CREATE_PLUGIN)GetProcAddress( lHandle, "createPlugin" );
	if ( ! lFunc )
	{
		// Handle the error
		FreeLibrary( lHandle );

		// return SOME_ERROR_CODE;
		return false;
	}
	
#else

    char* lError;

    //  cout << "\tTRY dlopen() : " << pFilename.c_str() << endl;
    void* lHandle = dlopen( pFilename.c_str(), RTLD_LAZY );
    if ( lHandle == NULL )
    {
		lError = dlerror();	
		if ( lError != NULL )
		{
	  		cout << lError << endl;
		}

        return false;
    }

    //    cout << "\tTRY dlsym( createPlugin )" << endl;
    dlerror();	//** Clear any existing error
   // GVV_CREATE_PLUGIN lFunc;
			// double (*cosine)(double);
			/* Writing: cosine = (double (*)(double)) dlsym(handle, "cos");
              would seem more natural, but the C99 standard leaves
              casting from "void *" to a function pointer undefined.
              The assignment used below is the POSIX.1-2003 (Technical
              Corrigendum 1) workaround; see the Rationale for the
              POSIX specification of dlsym(). */

  //  GVV_CREATE_PLUGIN lFunc = (GVV_CREATE_PLUGIN)dlsym( lHandle, "createPlugin" );
    *(void **) (&lFunc) = dlsym( lHandle, "createPlugin" );
	lError = dlerror();	
	if ( lError != NULL )
	{
	        cout << lError << endl;
	}
    if ( lFunc == NULL )
	{
		// cout << "\tSymbol createPlugin introuvable..." << endl;

		return false;
    }

#endif

	//** Call the function
    //    cout << "\tTRY createPlugin()" << endl;
    GvvPluginInterface* lPlugin = lFunc( *this );
    if ( lPlugin == NULL )
    {
        return false;
    }
    cout << "Plugin loaded : " << lPlugin->getName().c_str() << endl;

	//** Store plugin info
    PluginInfo lInfo;
    lInfo.mPlugin = lPlugin;
    lInfo.mHandle = lHandle;
    mPlugins.push_back( lInfo );

    return true;
}

/******************************************************************************
 * unloadAll()
 ******************************************************************************/
void GvvPluginManager::unloadAll()
{
    vector< PluginInfo >::iterator it;
    for ( it = mPlugins.begin(); it != mPlugins.end(); ++it )
    {
        PluginInfo& lPluginInfo = *it;

        delete lPluginInfo.mPlugin;
		lPluginInfo.mPlugin = NULL;

        //** Unload library
#ifdef _WIN32
		FreeLibrary( lPluginInfo.mHandle );
#else
		dlerror();    //** Clear any existing error
		dlclose( lPluginInfo.mHandle );
		char* lError = dlerror();
		if ( lError != NULL )
		{
			cout << lError << endl;
		}
#endif
	}

    mPlugins.clear();
}

/******************************************************************************
 * getNbPlugins()
 *
 * @return
 ******************************************************************************/
size_t GvvPluginManager::getNbPlugins() const
{
    return mPlugins.size();
}

/******************************************************************************
 * getNbPlugins()
 *
 * @param pIndex
 *
 * @return
 ******************************************************************************/
GvvPluginInterface* GvvPluginManager::getPlugin( int pIndex )
{
    if ( pIndex < 0 || pIndex >= getNbPlugins() )
    {
        return NULL;
    }

    return mPlugins[ pIndex ].mPlugin;
}
