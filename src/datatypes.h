#ifndef DATATYPES_H
#define DATATYPES_H

#include <cmath>
#include <iostream>

using namespace std;


struct Vec2f {
   float x,y;
   Vec2f() {
		x = y = 0;
   }
   
   Vec2f(float x_, float y_) {
          x = x_;
          y = y_;
   }
   
   Vec2f add(Vec2f v_) {
        return Vec2f(x+v_.x, y+v_.y);
   }
   
   Vec2f sub(Vec2f v_) {
        return Vec2f(x-v_.x, y-v_.y);
   }
   
   Vec2f mult(float s_) {
        return Vec2f(x*s_, y*s_);
   }
   
   float dot(Vec2f v_) {
        return x*v_.x + y*v_.y;
   }
   
   float norm() {
         return sqrt(x*x + y*y);
   }
   
   Vec2f normalize() {
        float length = norm();
        return Vec2f(x / length, y / length);
   }
   
};


#endif