/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _SAMPLEVIEWER_H_
#define _SAMPLEVIEWER_H_

#include <GL/glew.h>

#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

#include "SampleCore.h"

class SampleViewer : public QGLViewer
{
public:
	SampleViewer();
	~SampleViewer();

protected:
	virtual void init();
	virtual void draw();
	virtual void resizeGL(int width, int height);
	virtual QSize sizeHint() const;
	virtual void keyPressEvent(QKeyEvent *e);

private:
	SampleCore *mSampleCore;
	qglviewer::ManipulatedFrame* mLight1;
};

#endif // !_SAMPLEVIEWER_H_
