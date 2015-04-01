/************************************************************************

Copyright (C) 2012 Eric Heitz (er.heitz@gmail.com). All rights reserved.

This file is part of Qtfe (Qt Transfer Function Editor).

Qtfe is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version.

Qtfe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Qtfe.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "Qtfe.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QFile>
#include <QFileDialog>
#include <QTextStream>
#include <QVBoxLayout>
#include <QPainter>
#include <QDomDocument>
#include <QDomElement>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
 * @param pParent parent widget
 * @param pFlags window flags
 ******************************************************************************/
Qtfe::Qtfe( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_channels()
,	_outputs()
,	_filename()
{
	// Setup the Ui
	setupUi( this );

	// Add a layout to add channels and outputs widget in the transfer function container (groupbox)
	QVBoxLayout* layout = new QVBoxLayout( _transferFunctionGroupBox );

	// Do connections
	QObject::connect( _savePushButton, SIGNAL( clicked() ), this, SLOT( onSaveButtonClicked() ) );
	QObject::connect( _saveAsPushButton, SIGNAL( clicked() ), this, SLOT( onSaveAsButtonClicked() ) );
	QObject::connect( _loadToolButton, SIGNAL( clicked() ), this, SLOT( onLoadButtonClicked() ) );
	QObject::connect( _quitPushButton, SIGNAL( clicked() ), this, SLOT( onQuitButtonClicked() ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Qtfe::~Qtfe()
{
	// Iterate through outputs
	for ( int i = 0; i < _outputs.size(); ++i )
	{
		delete _outputs[ i ];
	}

	// Iterate through channels
	for ( int i = 0; i < _channels.size(); ++i )
	{
		delete _channels[ i ];
	}
}

/******************************************************************************
 * Initialize the editor by loading data from file
 *
 * @pFilename the file in Qtfe format
 ******************************************************************************/
bool Qtfe::load( const QString& pFilename )
{
	// TO DO
	// Add a mecanism to check the format of the file while parsing
	// ...

	if ( pFilename.isEmpty() )
	{
		// LOG
		QString logMessage = tr( "Qtfe::load() - Filename is empty" );
		std::cout << logMessage.toLatin1().constData() << std::endl;

		return false;
	}

	// Update member variables
	_filename = pFilename;
	_filenameLineEdit->setText( _filename );

	// Try to open/create file
	QFile file( _filename );
	if ( ! file.open( QIODevice::ReadOnly ) )
	{
		// LOG
		QString logMessage = tr( "Qtfe::load() - Unable to open file : " );
		logMessage += pFilename;
		std::cout << logMessage.toLatin1().constData() << std::endl;

		return false;
	}

	// Load file data in memory
	QDomDocument doc;
	if ( ! doc.setContent( &file ) )
	{
		// LOG
		QString logMessage = tr( "Qtfe::load() - Unable to set content of file in QDomDocument : " );
		logMessage += pFilename;
		std::cout << logMessage.toLatin1().constData() << std::endl;

		// Close file
		file.close();

		return false;
	}
	file.close();

	// Block functionChanged() signal while loading
	blockSignals( true );

	// Iterate through channels and destroy them
	for ( int i = 0; i < _channels.size(); ++i )
	{
		delete _channels[ i ];
	}
	_channels.clear();

	// Iterate through outputs and destroy them
	for ( int i = 0; i < _outputs.size(); ++i )
	{
		delete _outputs[ i ];
	}
	_outputs.clear();

	// Traverse the XML document
	QDomElement root = doc.documentElement();
	QDomNode node = root.firstChild();
	while( ! node.isNull() )
	{
		QDomElement element = node.toElement();

		// Search for channels
		if ( element.tagName() == "Function" )
		{
			// Create a new channel
			addChannels( 1 );

			// Retrieve list of points of current channel
			QDomNodeList points = element.childNodes();

			// Set first and last points of newly created channel
			QtfeChannel* channel = _channels.back();
			channel->setFirstPoint( points.item( 0 ).toElement().attributeNode( "y" ).value().toDouble() );
			channel->setLastPoint( points.item( points.length() - 1 ).toElement().attributeNode( "y" ).value().toDouble() );

			// Iterate through points and add intermediate points to newly created channel
			for ( uint i = 1; i < points.length() - 1; i++ )
			{
				// Retrieve current point
				QDomNode point = points.item( i );

				// Retrieve x and y values
				qreal x = point.toElement().attributeNode( "x" ).value().toDouble();
				qreal y = point.toElement().attributeNode( "y" ).value().toDouble();
		
				// Add point in channel
				channel->insertPoint( QPointF( x, y ) );
			}
		}

		// Search for outputs
		if ( element.tagName() == "Output" )
		{
			// Create a new output
			QtfeOutput* output = new QtfeOutput( this );

			// Do bindings
			output->bindChannelToR( element.attributeNode( "R" ).value().toInt() );	
			output->bindChannelToG( element.attributeNode( "G" ).value().toInt() );
			output->bindChannelToB( element.attributeNode( "B" ).value().toInt() );
			output->bindChannelToA( element.attributeNode( "A" ).value().toInt() );
			
			// Store the output
			_outputs.push_back( output );
			
			// Add it to main widget
			_transferFunctionGroupBox->layout()->addWidget( output );

			// Do connection(s)
			QObject::connect( output, SIGNAL( outputBindingChanged() ), this, SLOT( onOutputBindingChanged() ) );
		}

		// Go to next sibling in order to traverse the XML document
		node = node.nextSibling();
	}

	// Unblock functionChanged() signal
	blockSignals( false );

	// Emit signal
	emit functionChanged();

	return true;
}

/******************************************************************************
 * Add new channels
 *
 * @param pNbChannels the number of new channels
 ******************************************************************************/
void Qtfe::addChannels( int pNbChannels )
{
	// Iterate through number of channels
	for ( int i = 0; i < pNbChannels; ++i )
	{
		// Create a channel and store it
		QtfeChannel* channel = new QtfeChannel( this );
		_channels.push_back( channel );

		// Add it to main widget
		_transferFunctionGroupBox->layout()->addWidget( channel );

		// Do connection(s)
		QObject::connect( channel, SIGNAL( channelChanged() ), this, SLOT( onChannelChanged() ) );
	}
}

/******************************************************************************
 * Add new RGBA outputs
 *
 * @param pNbOuputs the number of new outputs
 ******************************************************************************/
void Qtfe::addOutputs( int pNbOuputs )
{
	// Iterate through number of outputs
	for ( int i = 0; i < pNbOuputs; ++i )
	{
		// Create an output and store it
		QtfeOutput* output = new QtfeOutput( this );
		_outputs.push_back( output );

		// Add it to main widget
		_transferFunctionGroupBox->layout()->addWidget( output );

		// Do connection(s)
		QObject::connect( output, SIGNAL( outputBindingChanged() ), this, SLOT( onOutputBindingChanged() ) );
	}	
}

/******************************************************************************
 * Bind a channel to an output (-1 to disable the channel)
 *
 * @param pChannelIndex index of channel
 * @param pOutputIndex index of output
 ******************************************************************************/
void Qtfe::bindChannelToOutputR( int pChannelIndex, int pOutputIndex )
{
	if ( 0 <= pOutputIndex && pOutputIndex < _outputs.size() )
	{
		_outputs[ pOutputIndex ]->bindChannelToR( pChannelIndex );
	}
}

/******************************************************************************
 * Bind a channel to an output (-1 to disable the channel)
 *
 * @param pChannelIndex index of channel
 * @param pOutputIndex index of output
 ******************************************************************************/
void Qtfe::bindChannelToOutputG( int pChannelIndex, int pOutputIndex )
{
	if ( 0 <= pOutputIndex && pOutputIndex < _outputs.size() )
	{
		_outputs[ pOutputIndex ]->bindChannelToG( pChannelIndex );
	}
}

/******************************************************************************
 * Bind a channel to an output (-1 to disable the channel)
 *
 * @param pChannelIndex index of channel
 * @param pOutputIndex index of output
 ******************************************************************************/
void Qtfe::bindChannelToOutputB( int pChannelIndex, int pOutputIndex )
{
	if ( 0 <= pOutputIndex && pOutputIndex < _outputs.size() )
	{
		_outputs[ pOutputIndex ]->bindChannelToB( pChannelIndex );
	}
}

/******************************************************************************
 * Bind a channel to an output (-1 to disable the channel)
 *
 * @param pChannelIndex index of channel
 * @param pOutputIndex index of output
 ******************************************************************************/
void Qtfe::bindChannelToOutputA( int pChannelIndex, int pOutputIndex )
{
	if ( 0 <= pOutputIndex && pOutputIndex < _outputs.size() )
	{
		_outputs[ pOutputIndex ]->bindChannelToA( pChannelIndex );
	}
}

/******************************************************************************
 * Get the transfer function dimension (channels number)
 *
 * @return the nb of channels
 ******************************************************************************/
int Qtfe::dim() const
{
	return _channels.size();
}

/******************************************************************************
 * Evaluate channel at x in [0.0, 1.0].
 * Invalid channel or x values return 0.0
 *
 * @param pChannelIndex index of channel
 * @param pXValue x position where to evaluate the function
 *
 * @return the function transfer value at given position
 ******************************************************************************/
qreal Qtfe::evalf( int pChannelIndex, qreal pXValue ) const
{
	if ( 0 <= pChannelIndex && pChannelIndex < _channels.size() )
	{
		return _channels[ pChannelIndex ]->evalf( pXValue );
	}
	else
	{
		return 0.0;
	}
}

/******************************************************************************
 * Slot called when a channel has been modified.
 ******************************************************************************/
void Qtfe::onChannelChanged()
{
	// Iterate through number of outputs
	for ( int i = 0; i < _outputs.size(); ++i )
	{
		_outputs[ i ]->repaint();
	}

	// Emit signal
	emit functionChanged();
}

/******************************************************************************
 * Slot called when an output binding has been modified.
 ******************************************************************************/
void Qtfe::onOutputBindingChanged()
{
	// Emit signal
	emit functionChanged();
}

/******************************************************************************
 * Slot called when the Save button has been clicked.
 *
 * Serialize the transfer funtion in file.
 * Currently, format is XML document.
 ******************************************************************************/
void Qtfe::onSaveButtonClicked()
{
	// If filename is empty, do "Save As"
	if ( _filename.isEmpty() )
	{
		onSaveAsButtonClicked();

		// Exit to avoid infinite loop
		return;
	}

	QDomDocument doc;
	QDomElement MultiEditor;
	QFile file;
	QTextStream out;

	// Try to open/create file
	file.setFileName( _filename );
	if ( ! file.open( QIODevice::WriteOnly ) )
	{
		// TO DO
		// Handle error
		//...

		return;
	}
	out.setDevice( &file );

	MultiEditor = doc.createElement( "Qtfe" );
	doc.appendChild( MultiEditor );

	// Iterate through channels
	for ( int i = 0; i < _channels.size(); ++i )
	{
		// Create a Function tag
		QDomElement func = doc.createElement( "Function" );
		MultiEditor.appendChild( func );

		// Iterate through points
		for ( int j = 0; j < _channels[ i ]->getPoints().size(); ++j )
		{
			// Create a point tag in Function tag
			QDomElement point = doc.createElement( "point" );
			func.appendChild( point );

			// Write data
			point.setAttribute( "x", QString::number( _channels[ i ]->getPoints()[ j ]->x() ) );
			point.setAttribute( "y", QString::number( _channels[ i ]->getPoints()[ j ]->y() ) );
		}
	}

	// Iterate through outputs
	for ( int i = 0; i < _outputs.size(); ++i )
	{
		// Create an Output tag
		QDomElement output = doc.createElement( "Output" );
		MultiEditor.appendChild( output );

		// Write data
		output.setAttribute( "R",  QString::number( _outputs[ i ]->_R ) );
		output.setAttribute( "G",  QString::number( _outputs[ i ]->_G ) );
		output.setAttribute( "B",  QString::number( _outputs[ i ]->_B ) );
		output.setAttribute( "A",  QString::number( _outputs[ i ]->_A ) );
	}
	
	// Add XML header at beginning of the file
	QDomNode noeud = doc.createProcessingInstruction( "xml", "version=\"1.0\"" );
	doc.insertBefore( noeud, doc.firstChild() );

	// Write document in file
	doc.save( out, 2 );

	// Close file
	file.close();
}

/******************************************************************************
 * Slot called when the Save As button has been clicked.
 *
 * Open a user interface dialog to save data in file.
 * Currently, format is XML document.
 ******************************************************************************/
void Qtfe::onSaveAsButtonClicked()
{
	// Open a user interface dialog to save file
	QString name = QFileDialog::getSaveFileName( 0, QString(), QString(), "*.xml" );
	if ( name.isEmpty() )
	{
		return;
	}
	
	// Update member variables
	_filename = name;
	_filenameLineEdit->setText( _filename );

	// Serialize data in file
	onSaveButtonClicked();
}

/******************************************************************************
 * Slot called when the Load button has been clicked.
 *
 * Open a user interface dialog to load a transfer function.
 * Currently, format is XML document.
 ******************************************************************************/
void Qtfe::onLoadButtonClicked()
{
	// TO DO
	// Add a mecanism to check the format of the file while parsing
	// ...

	// Open a user interface dialog to load file
	QString name = QFileDialog::getOpenFileName( 0, QString(), QString(), "*.xml" );
	if ( name.isEmpty() )
	{
		return;
	}

	// Update member variables
	_filename = name;
	_filenameLineEdit->setText( _filename );

	// Try to open/create file
	QFile file( _filename );
	if ( ! file.open( QIODevice::ReadOnly ) )
	{
		// TO DO
		// Handle error
		//...

		return;
	}

	// Load file data in memory
	QDomDocument doc;
	if ( ! doc.setContent( &file ) )
	{
		// TO DO
		// Handle error
		//...

		file.close();

		return;
	}
	file.close();

	// Block functionChanged() signal while loading
	blockSignals( true );

	// Iterate through channels and destroy them
	for ( int i = 0; i < _channels.size(); ++i )
	{
		delete _channels[ i ];
	}
	_channels.clear();

	// Iterate through outputs and destroy them
	for ( int i = 0; i < _outputs.size(); ++i )
	{
		delete _outputs[ i ];
	}
	_outputs.clear();

	// Traverse the XML document
	QDomElement root = doc.documentElement();
	QDomNode node = root.firstChild();
	while( ! node.isNull() )
	{
		QDomElement element = node.toElement();

		// Search for channels
		if ( element.tagName() == "Function" )
		{
			// Create a new channel
			addChannels( 1 );

			// Retrieve list of points of current channel
			QDomNodeList points = element.childNodes();

			// Set first and last points of newly created channel
			QtfeChannel* channel = _channels.back();
			channel->setFirstPoint( points.item( 0 ).toElement().attributeNode( "y" ).value().toDouble() );
			channel->setLastPoint( points.item( points.length() - 1 ).toElement().attributeNode( "y" ).value().toDouble() );

			// Iterate through points and add intermediate points to newly created channel
			for ( uint i = 1; i < points.length() - 1; i++ )
			{
				// Retrieve current point
				QDomNode point = points.item( i );

				// Retrieve x and y values
				qreal x = point.toElement().attributeNode( "x" ).value().toDouble();
				qreal y = point.toElement().attributeNode( "y" ).value().toDouble();
		
				// Add point in channel
				channel->insertPoint( QPointF( x, y ) );
			}
		}

		// Search for outputs
		if ( element.tagName() == "Output" )
		{
			// Create a new output
			QtfeOutput* output = new QtfeOutput( this );

			// Do bindings
			output->bindChannelToR( element.attributeNode( "R" ).value().toInt() );	
			output->bindChannelToG( element.attributeNode( "G" ).value().toInt() );
			output->bindChannelToB( element.attributeNode( "B" ).value().toInt() );
			output->bindChannelToA( element.attributeNode( "A" ).value().toInt() );
			
			// Store the output
			_outputs.push_back( output );
			
			// Add it to main widget
			//this->layout()->addWidget( output );
			_transferFunctionGroupBox->layout()->addWidget( output );

			// Do connection(s)
			QObject::connect( output, SIGNAL( outputBindingChanged() ), this, SLOT( onOutputBindingChanged() ) );
		}

		// Go to next sibling in order to traverse the XML document
		node = node.nextSibling();
	}

	// Unblock functionChanged() signal
	blockSignals( false );

	// Emit signal
	emit functionChanged();
}

/******************************************************************************
 * Slot called when the Quit button has been clicked.
 ******************************************************************************/
void Qtfe::onQuitButtonClicked()
{
	close();
}
