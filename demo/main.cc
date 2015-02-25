

#include <windows.h>
#include <iostream>
#include <GL/gl.h>
#include <GL/glut.h>
#include <math.h>
#include "../MultilayerPerceptron.h"

#define PI 3.1415926535

using namespace std;


int width = 640;
int height = 480;

int textureWidth = 64;
int textureHeight = 48;

unsigned char* img;
GLuint tex_image;
int iteration = 0;
int mode = 1;
bool training = false;

MultilayerPerceptron *mlp;
vector<MultilayerPerceptron::TrainingElement> trainingSet;





int main(int argc, char ** argv);
void display();
void reshape(int width, int height);
void mouse(int button, int state, int mx, int my);
void keyboardDown(unsigned char key, int x, int y);
void createImage(GLuint* texture_);
void addPointToTrainingSet(float x_, float y_);




int main(int argc, char ** argv) {

	img = (unsigned char*) malloc(textureWidth * textureHeight * 3 * sizeof(unsigned char));

	mlp = new MultilayerPerceptron(2, 3);
	mlp->addHiddenLayer(10);
	mlp->addHiddenLayer(10);
	mlp->init();
	mlp->setTrainingSet(trainingSet);

	glutInit(&argc, argv);
	glutInitWindowSize(width, height);
    glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow ("artificial neural network demo");
    glEnable(GL_DEPTH_TEST);
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
    glutMouseFunc(mouse);
	glutKeyboardFunc(keyboardDown);
	glutMainLoop();

	return 0;
}







void display(void) {

	iteration++;

	if(training) {
		float err = mlp->train(0.2f);
		createImage(&tex_image);
		cout << "iteration: " << iteration << ", error: " << err << "\n";
	}

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	MultilayerPerceptron::TrainingElement *te;
	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for(int k=0; k<trainingSet.size(); ++k) {
		te = &(trainingSet[k]);
		glColor3f(te->out[0], te->out[1], te->out[2]);
		glVertex2f(320*(1+te->in[0]), 240*(1+te->in[1]));
	}
	glEnd();


	glColor3f(1.0f, 1.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex_image);

    glPushMatrix();
        glTranslatef(0, 0, 0);
        glBegin(GL_QUADS);

        	glTexCoord2f(0.0, 0.0);
        	glVertex3f(0, 0, 0);

        	glTexCoord2f(1.0, 0.0);
        	glVertex3f(width, 0, 0);

        	glTexCoord2f(1.0, 1.0);
        	glVertex3f(width, height, 0);

        	glTexCoord2f(0.0, 1.0);
        	glVertex3f(0, height, 0);

        glEnd();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
	glutPostRedisplay();
}








void reshape(int width, int height) {
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D(0, width, height, 0);
	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}



void mouse(int button, int state, int x, int y) {
   switch( button ) {
       case GLUT_LEFT_BUTTON:
        	if( state == GLUT_UP )  {
				addPointToTrainingSet(x, y);
			}
			break;
	}
}


void createImage(GLuint* texture_) {

	vector<float> testInput;
	vector<float> testOutput;
	testInput.push_back(0.0);
	testInput.push_back(0.0);

	int ix, iy;
	for(iy = 0; iy < textureHeight; ++iy) {
		for(ix = 0; ix < textureWidth; ++ix) {
			testInput[0] = (2.0f*ix/textureWidth)-1;
			testInput[1] = (2.0f*iy/textureHeight)-1;
			testOutput = mlp->classify(testInput);
			img[3*(iy*textureWidth+ix)+0] = 128 * testOutput[0];
			img[3*(iy*textureWidth+ix)+1] = 128 * testOutput[1];
			img[3*(iy*textureWidth+ix)+2] = 128 * testOutput[2];
		}
	}

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures( 1, texture_ );
	glBindTexture( GL_TEXTURE_2D, *texture_ );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, &img[0]);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

}


void addPointToTrainingSet(float x_, float y_) {

	float x = (x_-0.5*width) / (0.5 * width);
	float y = (y_-0.5*height) / (0.5 * height);

	vector<float> teInput;
	teInput.push_back(x);
	teInput.push_back(y);
	vector<float> teOutput;
	teOutput.push_back(0.0);
	teOutput.push_back(0.0);
	teOutput.push_back(0.0);
	teOutput[mode-1] = 1.0f;

	trainingSet.push_back(MultilayerPerceptron::TrainingElement(teInput, teOutput));
	mlp->setTrainingSet(trainingSet);
}


void keyboardDown(unsigned char key, int x, int y) {
    switch (key) {
      case '1':
      {
           mode = 1;
           break;
      }
      case '2':
      {
			mode = 2;
            break;
      }
      case '3':
      {
			mode = 3;
            break;
      }
      case '4':
	  {
			training = true;
			break;
	  }
	  case '5':
	  {
			training = false;
			mlp->resetWeights();
			trainingSet.clear();
			createImage(&tex_image);
			break;
	  }
    }
}