/* icp_matcher.i */



%module icp_matcher
 %{
 /* Put header files here or function declarations like below */
 #include "icp_matcher.hpp"
 %}


%include "std_vector.i"


// Instantiate templates
namespace std {
   %template(IntVector) vector<int>;
   %template(FloatVector) vector<float>;
   %template(FloatVectorVector) vector< vector<float> >;
}

// Include the header file with above prototypes
%include "icp_matcher.hpp"
