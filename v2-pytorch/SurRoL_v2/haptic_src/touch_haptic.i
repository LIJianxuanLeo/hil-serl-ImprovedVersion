%module touch_haptic


%{
#define SWIG_FILE_WITH_INIT
#include "INCLUDE/touch_haptic.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* retrived_info, int n1)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* retrived_info2, int n2)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* pose_info, int n3)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* pose_info2, int n4)}

int initTouch_right();
int initTouch_left();
void startScheduler();
void stopScheduler();
void closeTouch_left();
void closeTouch_right();
void getDeviceAction_right(float* retrived_info, int n1);
void getDeviceAction_left(float* retrived_info2, int n2);
void getDevicePose_right(float* pose_info, int n3);
void getDevicePose_left(float* pose_info2, int n4);


#include "INCLUDE/touch_haptic.h"
