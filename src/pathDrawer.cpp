#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>


#include <fstream>
#include <sstream>
#include <cstdio>

#include <X11/Xlib.h>

#include "pathFinder.hpp"
#include "../include/CVT.h"

#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

extern "C" {
#include "../include/dbscan.h"
}
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat resized, img1, img, imgFiltered, imgOriginal;

Point pointROI, pointResized;
vector<Point> pts, ptsResized;
std::vector<Point2f> pointsFiltered;
int drag = 0;
int var = 0;
int flag = 0;
double convFactor=1.0;

bool parserCommand(int argc, char ** argv, Mat &img , int &N, int &E,int &pointSize, int &lineSize, bool &drawRepeat, bool &drawPath, bool &onlyPath, int &horSeg, int &verSeg) {
	const String keys =
	    "{help h usage ? |      | print this message      }"
	    "{@image         |      | image for stippling     }"
	    "{N n number     |      | points amount (-N= -n= , integer value}"
	    "{inverse i      |      | inverse image, usually needed to get better results}"
	    "{epoch e        |      | epochs (-e= -epoch=, integer value}"
	    "{pointSize ps   |      | point size (-ps= or -pointSize= , integer value)}"
	    "{lineSize ls    |      | line size (-ls= or -lineSize= , integer value)}"
	    "{draw d         |      | show iterate processing (proccess of stippling)}"
	    "{drawPath dp  |      | show the path drawing process }"
	    "{onlyPath op  |      | skips the stippling process }"
	    "{horizontalSeg horSeg  |      | Horizontal segmentation of the image }"
		"{verticalSeg verSeg  |      | Vertical segmentation of the image }"
	    ;

	CommandLineParser parser(argc, argv, keys);
	parser.about("Weighted Voronoi Redering.");
	if (parser.has("help")) {
		parser.printMessage();
		return false;
	}

	// Open Image.
	String imgPath = parser.get<String>(0);
	if (imgPath.empty() && !parser.has("onlyPath")) {
		parser.printMessage();
		return false;
	}

	img = imread(imgPath);
	if (img.empty() && !parser.has("onlyPath")) {
		cout << "Error Loading Image" << endl;
		return false;
	}

	if(!img.empty()){
	imgOriginal=img.clone();
	cvtColor(img, img, COLOR_BGR2GRAY);
	const Size size = img.size();
	// Set Points Number.
		N = (size.height + size.width) * 2;
			if (parser.has("n"))
				N = parser.get<int>("n");
	}		

	// Inverse.
	if (parser.has("inverse"))
		img = ~img;

	// Set Epochs Number.
	E = 100;
	if (parser.has("epoch"))
		E = parser.get<int>("epoch");
	
	// Set Point Size.
	pointSize = 1;
	if (parser.has("pointSize"))
		pointSize = parser.get<int>("pointSize");

	// Set Line Size.
	lineSize = 1;
	if (parser.has("lineSize"))
		lineSize = parser.get<int>("lineSize");

	// Show Stippling Processing.
	drawRepeat = false;
	if (parser.has("draw"))
		drawRepeat = true;

	// Show Path Process
	drawPath = false;
	if (parser.has("drawPath"))
		drawPath = true;

	// It skips Stippling Processing.
	onlyPath = false;
	if (parser.has("onlyPath"))
		onlyPath = true;
	
	// Set Horizontal Segmentation.
	horSeg = 100;
	if (parser.has("horizontalSeg"))
		horSeg = parser.get<int>("horizontalSeg");	
	
	// Set Vertical Segmentation
	verSeg = 100;
	if (parser.has("verticalSeg"))
		verSeg = parser.get<int>("verticalSeg");

	// Check Parser Error.
	if (!parser.check()) {
		parser.printErrors();
		return false;
	}

	return true;
}


void appendRandomPoint(RNG &rng, vector<Point2f> &points, Size size, int N) {
	int validPointsCounter = 0;
	while(validPointsCounter<N){
			float x = rng.uniform((float)0, (float)size.width - 1);
			float y = rng.uniform((float)0, (float)size.height - 1);
			if(pointPolygonTest(pts, Point2f(x, y), false)>=0){
				points.push_back(Point2f(x, y));
				validPointsCounter++;
			}
	}
}

//Method to check during reading the stippled_points.txt file in case the stippling process was skipped (-op (or -onlyPath) option) that the give image as outpupt is big enough to contain all
//the input points
bool pointInsideImage(int x2, int y2, double x, double y) 
{ 
    if (x > 0 and x < x2 and y > 0 and y < y2) 
        return true; 
  
    return false; 
} 



void mouseHandler(int, int, int, int, void*);

void mouseHandler(int event, int x, int y, int, void*)
{

    if (event == EVENT_LBUTTONDOWN && !drag)
    {
        if (flag == 0)
        {
            if (var == 0)
                img1 = resized.clone();
            pointROI = Point(x*convFactor, y*convFactor);
			pts.push_back(pointROI);
			pointResized = Point(x, y);
            circle(img1, pointResized, 2, Scalar(0, 0, 255), -1, 8, 0);
            ptsResized.push_back(pointResized);
            var++;
            drag  = 1;

            if (var > 1)
                line(img1,ptsResized[var-2], pointResized, Scalar(0, 0, 255), 2, 8, 0);

            imshow("Source", img1);
        }
    }

    if (event == EVENT_LBUTTONUP && drag)
    {
        imshow("Source", img1);
        drag = 0;
    }

    if (event == EVENT_RBUTTONDOWN)
    {
        flag = 1;
        img1 = img.clone();

        if (var != 0)
        {
            polylines( img1, pts, 1, Scalar(0,0,0), 2, 8, 0);
        }

//        imshow("Source", img1);
    }

    if (event == EVENT_RBUTTONUP)
    {
        flag = var;
		destroyAllWindows();
}

    if (event == EVENT_MBUTTONDOWN)
    {
        pts.clear();
		ptsResized.clear();
        var = 0;
        drag = 0;
        flag = 0;
        imshow("Source", resized);
    }
}


int main(int argc, char ** argv) {

//Beggining of the code for the stippling
	RNG rng(time(0));
	int N, E, pointSize, lineSize, horSeg, verSeg;
	bool drawRepeat, drawPath, onlyPath;
	ofstream myfile;
	Mat imgPath;
	std::vector<Point2f> solutionPath;
	std::vector<double> coords;

	if (!parserCommand(argc, argv, img, N, E, pointSize, lineSize, drawRepeat, drawPath, onlyPath, horSeg, verSeg )) {
		return -1;
	}
	
	// Point Set.
	vector<Point2f> points;

if(!onlyPath){
	//First the program allows the user to define the interest region within the image, so the background can be ignored and 
	//no points will be created there. The stippling only takes place in this region
	
	 cout << "\n\tleft mouse button - set a point to create mask shape\n"
    "\tright mouse button - create mask from points\n"
    "\tmiddle mouse button - reset\n";
	
    if( img.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }

 	double horizontalRes = 0;
	double verticalRes = 0;
	Display* d = XOpenDisplay(NULL);
	Screen*  s = DefaultScreenOfDisplay(d);
	verticalRes=s->height;
	horizontalRes=s->width;
   	
	if((img.rows>verticalRes) || (img.cols>horizontalRes)){
		convFactor = 1.1*(std::max((img.rows/verticalRes), (img.cols/horizontalRes)));

		resize(imgOriginal, resized, Size(), 1/convFactor, 1/convFactor, INTER_LANCZOS4);
	}
	else{resized=imgOriginal;}
	
    namedWindow("Source", WINDOW_AUTOSIZE);
    setMouseCallback("Source", mouseHandler, NULL);
    imshow("Source", resized);
    waitKey(0);
		
			
	if( remove( "stippled_points.txt" ) != 0 )
 		perror( "Error deleting stippled_points.txt:" );
 	else
   		puts( "File stippled_points.txt successfully deleted" );

   	puts( "Stippling starts \n" );

	const Size size = img.size();


	// To avoid the Density of dark region being too small.
	img = img * 254 / 255 + 1;


	// Add Points.
	appendRandomPoint(rng, points, size, 200);

	Mat imgVoronoi(size.height, size.width, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < E; ++i) {

		// Append Random Points.
		if (i < (N / 200))
			appendRandomPoint(rng, points, size, 200);

		// Remove Out of range Points.
		for (auto i = points.begin(); i != points.end(); ++i) {
			if (i->x > size.width || i->x < 0 || i->y > size.height || i->y < 0) {
				points.erase(i);
				i--;
			}
		}

		// Subdivision.
		Rect rect(0, 0, size.width, size.height);
		Subdiv2D subdiv(rect);
		subdiv.insert(points);

		// Subdiv Mat.
		imgVoronoi = Scalar(255, 255, 255);
		points = CVT::drawVoronoi(img, imgVoronoi, subdiv, pointSize);

		// Show Image.
		if (drawRepeat) {
			imshow( "imgVoronoi", imgVoronoi);
			waitKey(10);
		}

		cout << "(" << i << "/" << E << ")" << endl;

	}
	
cout << endl << "done." << endl;

destroyAllWindows();

cout << endl << "Proceding with the filtering: " << endl;

point_t *pointsDBSCAN;
double epsilon=30;
unsigned int minpts=25;
unsigned int num_points = points.size();
unsigned int m = 0;
imgFiltered=imgOriginal.clone();
imgFiltered.setTo(cv::Scalar(255,255,255));

//Filtering with DBSCAN and interactive input of the user for the parameters
while(1) {

cout<< "\nEnter the epsilon value (the radius to look for points around). It has to be a positive number! \n";
cin>>epsilon;

while(1)
{
if(cin.fail())
{
cin.clear();
cin.ignore(numeric_limits<streamsize>::max(),'\n');
cout<<"You have entered wrong input. Try again to enter the epsilon value (positive number)"<<endl;
cin>>epsilon;
}
if(!cin.fail() && epsilon>0)
{
cout<<"\n";
break;
}
}

cout<<"\nEnter the minimum number of points required to be considered a cluster (positive integer)\n";
cin>>minpts;
while(1)
{
if(cin.fail())
{
cin.clear();
cin.ignore(numeric_limits<streamsize>::max(),'\n');
cout<<"You have entered wrong input. Try again to enter the minimum number of points required to be considered a cluster (positive integer)"<<endl;
cin>>minpts;
}
if(!cin.fail() && minpts>0)
{
cout<<"\n";
break;
}
}

m=0;

point_t *p = (point_t *) calloc(num_points, sizeof(point_t));

if (p == NULL) {
    perror("Failed to allocate points array");
    return 0;
}

    while (m < num_points)
	{
     p[m].x=points[m].x;
	 p[m].y=points[m].y;
	 p[m].z=1.0;
      p[m].cluster_id = UNCLASSIFIED;    
	  ++m;
	 }
		  
pointsDBSCAN = p;
	
dbscan(pointsDBSCAN, num_points, epsilon, minpts, euclidean_dist);

int s=0;
pointsFiltered.clear();

    while (s <= num_points) {
		if(pointsDBSCAN[s].cluster_id >= 0){
            pointsFiltered.push_back(Point2f(pointsDBSCAN[s].x, pointsDBSCAN[s].y));
		}
          ++s;
    }

imgFiltered=imgOriginal.clone();
imgFiltered.setTo(cv::Scalar(255,255,255));

 	for(int i = 0; i < pointsFiltered.size(); i++) {
	circle(imgFiltered, pointsFiltered[i],  1, Scalar(0, 0, 0), -1, 8, 0);
    }

Mat imgFilteredResize = imgFiltered.clone();

resize(imgFiltered, imgFilteredResize, Size(), 1/convFactor, 1/convFactor, INTER_LANCZOS4);
	
imshow("filteredImage", imgFilteredResize);
waitKey(0);

destroyAllWindows();

int useImgFiltered=0;
std::cout<<"\nDo you want to use these points? 1 if yes, 0 if no"<<std::endl;
cin>>useImgFiltered;

while(1){
if(cin.fail())
{
cin.clear();
cin.ignore(numeric_limits<streamsize>::max(),'\n');
cout<<"You have entered wrong input. Do you want to use these points? 1 if yes, 0 if no"<<endl;
cin>>useImgFiltered;
}
if(!cin.fail() && (useImgFiltered==1 || useImgFiltered==0))
{
cout<<"\n";
break;
}
}

if(useImgFiltered==1)
	break;
}//End of the interactive filtering

	imwrite("filteredImage.jpg", imgFiltered);

	imwrite("stippled_image.jpg", imgVoronoi);

	myfile.open ("stippled_points_noFiltered.txt");

	for (auto &Point2f : points) // access by reference to avoid copying
	{  
	//Checks that the actual value obtained it is a point within the image and not outside (there can appear some very big numbers due to numerical singularities)
		if(Point2f.x < size.width && Point2f.y<size.height ){
			myfile << Point2f.x << " " << Point2f.y << "\n";
		}
	}

	myfile.close();
	
		for (auto &Point2f : points) // access by reference to avoid copying
	{  
	//Checks that the actual value obtained it is a point within the image and not outside (there can appear some very big numbers due to numerical singularities)
		if(Point2f.x < size.width && Point2f.y<size.height ){
			myfile << Point2f.x << " " << Point2f.y << "\n";
		}
	}

	myfile.close();
	
	myfile.open ("stippled_points.txt");

std::puts( "Stippling done!");
std::puts( "The created file stippled_points.txt contains the (x,y) coordinates of the stippling points, and stippled_image.jpg was also created");
}
else
{
std::puts( "Stippling skipped! Proceding directly to the path finding of the input points contained in the file given by the user stippled_points.txt" );
}
//End of the code for the stippling



//Beggining of the code correspoding to the path finding

puts( "\nPath finding starts: \n" );

//Normal processing
if (!onlyPath){
Size size = img.size();
// Creates the image that will be given as path finding output
imgPath = Mat(size.height, size.width, CV_32F);

//The default process is with the filtered points
points=pointsFiltered;
}
//Walkaround in case we only wanted to do the path, skipping the stippling process
else{
int height;
int width;
bool imageBigEnough=false;

std::ifstream file("stippled_points.txt");
std::string str;

//Converting the original pair of coordinates obtained by the stippling into a vector of non-differenciated-by-points coordinates that Delanuator takes as input (coords), reading the from a file called 
//stippled_points.txt with two columns with the format "xCoordinate yCoordinate" and each row being one point- In this code this would not be required and we could obtain directly "points",
//but I am reusing code and it works well so I don't modify it.

std::string   fileLine;
int linesCounter=0;
while(std::getline(file, fileLine))
{
linesCounter++;
        std::stringstream  lineStream(fileLine);

        double value;
        while(lineStream >> value)
        {
            coords.push_back(value);
        }
}

//Checks that the file was correctly read
if(coords.size()/2 != linesCounter)
{
cout << "There is an error in your input file stippled_points.txt and not all the points were well read. Please check the file and try again!\n \n \n";
return 0;
}

//The previously read coordinates are transformed into pairs of points
int counter=0;

for(int i=0; i<coords.size(); i++){
	counter++;
	if(counter==2){
		Point2f point(coords[i-1],coords[i]);
		points.push_back(point);
		counter=0;
	}
}


while(1) {

cout<< "\nEnter the height of the output image (integer number of pixels)\n";
cin>>height;

while(1)
{
if(cin.fail())
{
cin.clear();
cin.ignore(numeric_limits<streamsize>::max(),'\n');
cout<<"You have entered wrong input. Try again to enter the height of the output image (integer number of pixels)"<<endl;
cin>>height;
}
if(!cin.fail())
{
cout<<"\n";
break;
}
}

cout<<"\nEnter the width of the output image (integer number of pixels)\n";
cin>>width;
while(1)
{
if(cin.fail())
{
cin.clear();
cin.ignore(numeric_limits<streamsize>::max(),'\n');
cout<<"You have entered wrong input. Try again to enter the width of the output image (integer number of pixels)"<<endl;
cin>>height;
}
if(!cin.fail())
{
cout<<"\n";
break;
}
}

for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
{
Point2f pointAux = *it;
if (!pointInsideImage(width, height, pointAux.x, pointAux.y))
{
		cout << "The size of the image you gave is not big enough to contain the points. Please try again!\n \n \n \n \n";
		imageBigEnough=false;
		break;
}
else
	imageBigEnough=true;

}

if(imageBigEnough)
	break;
 
}

imgPath = Mat(height, width, CV_32F);
}
//End of the walkaround in case we want to skip the stippling process

//The background color for the image is set here
imgPath.setTo(cv::Scalar(255,255,255));
Size size = imgPath.size();

//The color of the lines is set by path_colorLines
Scalar path_colorLines(0,0,0);

string win_path = "Path dynamic window";

//Path finding happens here
PathFinder::pathFinder pathCalculation(points, imgPath, horSeg, verSeg);
solutionPath = pathCalculation.getPath();

std::puts( "Path finding done!" );

//Here it draws the image with the delanuy tesselation and creates a file "triangles.txt" containing the triangles generated
if( std::remove( "path_finding.txt" ) != 0 )
	std::perror( "Error deleting previous path_finding.txt file (not found)" );
 else
	std::puts( "Previous pathFinding.txt successfully deleted" );

std::ofstream myFilePathFinding;
myFilePathFinding.open ("path_finding.txt");

	for(int i = 0; i < solutionPath.size(); i++) {
	myFilePathFinding << solutionPath[i].x << " " << solutionPath[i].y << "\n";
    }
	
	for(int i = 0; i < solutionPath.size(); i++) {
		line(imgPath, solutionPath[i], solutionPath[i+1], path_colorLines, lineSize, CV_8S, 0);
		if (drawPath)
    	{
			imshow(win_path, imgPath);
			waitKey(1);
    	}
	}


myFilePathFinding.close();

//Writes the path image
imwrite( "Path_Image.jpg", imgPath);

std::puts( "Path drawing done!" );

return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif