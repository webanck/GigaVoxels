/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

#include "ShaderManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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

//#define RESET   "\033[0m"
//#define RED     "\033[31m"      /* Red */

int *nbLignes = NULL;

/******************************************************************************
 * ...
 ******************************************************************************/
pair< char**, int > loadSource( const char* filename )
{
	vector< char* > includedFiles;	
	char** src = NULL;	//toutes les sources
	char* shadersrc = NULL;	//code source du shader lui-même
	FILE* fp = NULL;	//fichier shader
	FILE* tf = NULL;	//fichier output pour tester
	long size;

	fp = fopen( filename, "rb" );
	tf = fopen( "shaderOutput.txt", "w" );
	if ( fp == NULL )
	{
		cout << "Echec d'ouverture du fichier." << endl;
	}

	// Récupérer la longueur du fichier
	fseek( fp, 0, SEEK_END );
	size = ftell( fp );

	/* On se replace au debut du fichier */
	rewind( fp );

	/* On alloue de la memoire pour y placer notre code source */
	shadersrc = (char*)malloc( size + 1 ); /* +1 pour le caractere de fin de chaine '\0' */
	if ( shadersrc == NULL )
	{
		fclose( fp );
		cerr << "Erreur d'allocation de memoire!" << endl;

	}
	/* lecture du fichier */
	char c;    
	char str[ 8 ];
	char* file = NULL;
	int i = 0;
	int pos;
	bool commstar = false;//si on est dans un commentaire /*...*/
	bool commsimple = false;//si on est dans un commentaire //
	while ( i < size )
	{
		c = fgetc( fp ); 	//on récupère le caractère

		// Pre-processor "include files" Managemnt
		/*if (c == '#') {				
		fgets(str, 8, fp); //on lit les 7 prochains caractères
		if (strcmp(str, "include") == 0) {//si il y a #include
		size -= 8;//on diminue la taille de la string finale
		do {c = fgetc(fp); size--;} while (c == ' ');//tant qu'il y a des espaces, avancer		
		//début du nom du fichier includé (1er caractère lu)
		pos = ftell(fp)-1;//position du 1er caractère du nom du fichier
		int filelength = 1;
		do {
		c = fgetc(fp);
		filelength++;
		} while (c != '\n' && c != '*' && c != ' ');
		//longueur du nom du fichier récupéré
		size -= (filelength-2);
		fseek(fp, pos, SEEK_SET);//retour au début du nom du fichier
		file = new char[filelength];//on a le nom du fichier inclus						
		fgets(file, filelength, fp);
		if (!commsimple && !commstar) {//on ne l'ajoute que si on est pas dans un commentaire
		includedFiles.push_back(file);
		}
		c = fgetc(fp); //en general \n (pour conserver le même nombre de lignes)
		}
		} else if (c == '/') {//test pour savoir si on est dans un commentaire
		int p = ftell(fp);
		char d = fgetc(fp);
		if (d == '/') {
		commsimple = true;
		} else if (d == '*') {
		commstar = true;
		}
		fseek(fp, p, SEEK_SET);
		} else if (c == '*') {//test pour savoir si on doit fermer un commentaire
		int p = ftell(fp);
		char d = fgetc(fp);
		if (d == '/' && commstar) {
		commstar = false;
		}
		fseek(fp, p, SEEK_SET);
		} else if (c == '\n') {//retour chariot: on n'est plus dans un commentaire simple
		commsimple = false;
		}*/
		shadersrc[ i ] = c;	//on rajoute le caractère à la string
		fputc( c, tf );
		i++;
	}
	shadersrc[ size ] = '\0';

	fclose( tf );
	fclose( fp );

	int nbIncludes = includedFiles.size();

	// Création des sources
	src = new char* [ nbIncludes + 1 ];
	nbLignes = new int[ nbIncludes ];//tableau gardant le nombre de lignes de code par fichier inclus
	cout << nbIncludes << " " << "fichier(s) inclus dans le shader " << filename << ":" << endl;
	for ( int i = 0; i < nbIncludes; i++ )
	{
		FILE* iF = fopen( includedFiles[ i ], "r" );
		cout << includedFiles[ i ] << endl;
		if ( iF == NULL )
		{
			cout << "Echec d'ouverture du fichier inclus " << includedFiles[ i ] << "." << endl;
		}
		fseek( iF, 0, SEEK_END );
		int size = ftell( iF );
		rewind( iF );
		src[ i ] = (char*)malloc( size + 2 );
		int nbL = 1;
		for ( int k = 0; k < size; k++ )
		{
			src[ i ][ k ] = fgetc( iF );
			if ( src[ i ][ k ] == '\n' )
			{
				nbL++;
			}
		}
		nbLignes[ i ] = nbL;
		src[ i ][ size ] = '\n';
		src[ i ][ size + 1 ] = '\0';
		fclose( iF );
	} 
	src[ nbIncludes ] = shadersrc;	// On met le code du shader lui-même à la toute fin

	return make_pair( src, nbIncludes );
}

/******************************************************************************
 * ...
 ******************************************************************************/
GLuint useShader( GLenum shaderType, const char* filename )
{
	GLuint shader = 0;
	char** vsrc = NULL;
	GLsizei logsize = 0;
	GLint compile_status = GL_TRUE;
	char* log = NULL;

	shader = glCreateShader( shaderType );
	if ( shader == 0 )
	{
		cerr << "Erreur de création de shader." << endl;
		return 0;
	}
	pair< char**, int > pair;
	pair = loadSource( filename );
	vsrc = pair.first;
	int nbIncludes = pair.second;
	if ( vsrc == NULL )
	{
		glDeleteShader( shader );

		return 0;
	} 
	glShaderSource( shader, nbIncludes+1, (const char**)vsrc, NULL );
	glCompileShader( shader );

	glGetShaderiv( shader, GL_COMPILE_STATUS, &compile_status );
	if ( compile_status != GL_TRUE )
	{
		/* erreur a la compilation recuperation du log d'erreur */

		/* on recupere la taille du message d'erreur */
		glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

		/* on alloue un espace memoire dans lequel OpenGL ecrira le message */
		log = (char*)malloc( logsize + 1 );
		if ( log == NULL )
		{
			cerr << "Impossible d'allouer de la memoire !" << endl;

			return 0;
		}
		/* initialisation du contenu */
		memset( log, '\0', logsize + 1 );

		glGetShaderInfoLog( shader, logsize, &logsize, log );

		char* newlog = NULL;//au cas où des fichiers sont inclus: 
		//il faut changer les numéros de lignes où
		//il y a des erreurs

		if ( strcmp(log, "" ) != 0 )
		{
			if ( nbIncludes > 0 )
			{
				//calcul du nombre de lignes dans les fichiers inclus
				int nbLignesOffset = 0;
				for ( int i = 0; i < nbIncludes; i++ )
				{
					nbLignesOffset += nbLignes[ i ];
				}	     
				vector< int > firstPositions;// positions des '(' qui nous intéressent
				vector< int > secondPositions;//positions des ')' qui nous intéressent
				for ( int k = 0; k < logsize + 1; k++ )
				{
					if ( log[ k ] == '(' )
					{
						firstPositions.push_back( k );
					}
					if ( log[ k ] == ')' )
					{
						secondPositions.push_back( k );
					}
				}
				if ( firstPositions.size() != secondPositions.size() )
				{
					cout << "Bizarre: pas le même nombre de '(' que de ')' "<< endl;
				}

				vector< int > oldSizes; //nb de caractères dans les numéro de lignes de l'ancien log
				vector< string > newLines;//strings des nouveaux numéros de lignes à mettre
				vector< int > newSizes;//nb de caractères dans les numéro de lignes du nouveau log

				for ( int a = 0; a < firstPositions.size(); a++ )
				{			
					vector< char > oldLineNumber;
					int k = firstPositions[ a ] + 1;
					while ( log[ k ] != ')' && k < logsize + 1 )
					{
						oldLineNumber.push_back( log[ k ] );
						k++;
					}
					const int oldsize = oldLineNumber.size();
					oldSizes.push_back( oldsize );
					char* oldLNstr;
					oldLNstr = new char[ oldsize + 1 ];
					for ( int j = 0; j < oldsize; j++ )
					{
						oldLNstr[ j ] = oldLineNumber[ j ];
					}
					oldLNstr[ oldsize ] = '\0';//numero de ligne qu'il y avait avant dans le message d'erreur (str)
					int oldLN;//numero de ligne qu'il y avait avant dans le message d'erreur (int)
					stringstream ss( oldLNstr );
					ss >> oldLN;
					int newLN = oldLN - nbLignesOffset;//nouveau numero de ligne (int)
					string newLNstr;
					ostringstream convert;
					convert << newLN;
					newLNstr = convert.str();//nouveau numero de ligne (str)
					newLines.push_back( newLNstr );
					int newsize = newLNstr.size();
					newSizes.push_back( newsize );
					delete oldLNstr;
				}

				int sumOldSizes = 0;//sommes des anciens nb de caractères des numéros de ligne
				int sumNewSizes = 0;//sommes des nouveaux nb de caractères des numéros de ligne
				for ( int a = 0; a < oldSizes.size(); a++ )
				{
					sumOldSizes += oldSizes[ a ];
					sumNewSizes += newSizes[ a ];
				}

				//nouveau message d'erreur
				newlog = (char*)malloc( logsize + 1 - ( sumOldSizes - sumNewSizes ) );

				int beg = 0; //début du bout de string à traiter
				int end; //fin du bout de string à traiter
				if ( firstPositions.size() > 1 )
				{
					end = firstPositions[ 1 ] - 1;
				}
				else
				{
					end = logsize - ( sumOldSizes - sumNewSizes );
				}
				bool fini = false;//test fin de boucle
				int a = 0;
				while ( ! fini )
				{
					for ( int n = beg; n <= firstPositions[ a ]; n++ )
					{//jusqu'à la '('
						newlog[ n ] = log[ n ];
					}
					for ( int n = 1; n <= newSizes[ a ]; n++ )
					{//entre '(' et ')'
						newlog[ firstPositions[ a ] + n ] = newLines[ a ][ n - 1 ]; 
					}
					for ( int n = firstPositions[ a ] + newSizes[ a ] + 1; n <= end; n++ )
					{//à partir de ')'
						newlog[ n ] = log[ n + oldSizes[ a ] - newSizes[ a ] ];
					}
					if ( end == logsize - ( sumOldSizes - sumNewSizes ) )
					{// si on est à la fin
						fini = true;
					}
					a++;
					//mise à jour de beg et end
					beg = end;
					if ( firstPositions.size() > a + 1 )
					{
						end = firstPositions[ a + 1] - 1;
					}
					else
					{
						end = logsize - ( sumOldSizes - sumNewSizes );
					}
				}
				newlog[ logsize - ( sumOldSizes - sumNewSizes ) ] = '\0';
				cerr << "Impossible de compiler le shader '"<< filename << "' :" << endl << newlog << endl;
			}
			else
			{
				cerr << "Impossible de compiler le shader '"<< filename << "' :" << endl << log << endl;
			}
		}
		/* ne pas oublier de liberer la memoire et notre shader */
		free( log );
		glDeleteShader( shader );

		return 0;
	}
	return shader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void linkStatus( GLuint program )
{
	GLsizei logsize = 0;
	char* log = NULL;
	GLint link_status = GL_TRUE;

	glGetProgramiv( program, GL_LINK_STATUS, &link_status );
	if ( link_status != GL_TRUE )
	{
		/* erreur a la compilation recuperation du log d'erreur */

		/* on recupere la taille du message d'erreur */
		glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

		/* on alloue un espace memoire dans lequel OpenGL ecrira le message */
		log = (char*)malloc( logsize + 1 );
		if ( log == NULL )
		{
			cerr << "Impossible d'allouer de la memoire !" << endl;
			//return ;
		}
		/* initialisation du contenu */
		memset( log, '\0', logsize + 1 );

		glGetProgramInfoLog( program, logsize, &logsize, log );
		cerr << "Impossible de linker le programme :" << endl << log << endl;

		/* Ne pas oublier de liberer la memoire et notre programme */
		free( log );
		glDeleteProgram( program );

		//return ;
	}
	//return ;
} 
