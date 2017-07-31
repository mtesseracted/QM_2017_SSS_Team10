#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <vector>

namespace py = pybind11;


double dot_prod(const std::vector<double>& v1, const std::vector<double>& v2)
{
    int sz1 = v1.size(); int sz2 = v2.size();
    if(sz1 != sz2)
	throw std::runtime_error("Vector lengths do not match");
    if(sz1 == 0)
	throw std::runtime_error("Zero-length vector");

    double dot = 0.0;
    for(size_t i=0; i<sz1; i++){
	dot += v1[i]*v2[i];
    }
    return dot;
}

//uncasted start method
//double dot_prod_np(py::array v0, py::array v1)
//cast explicitly
double dot_prod_np(py::array_t<double> v0,
                   py::array_t<double> v1)
{
    py::buffer_info v1_info = v1.request();
    py::buffer_info v0_info = v0.request();

    if(v0_info.ndim != 1)
	throw std::runtime_error("v0 is not a vector");
    if(v1_info.ndim != 1)
	throw std::runtime_error("v1 is not a vector");
    if(v0_info.shape[0] != v1_info.shape[0])
	throw std::runtime_error("v0 and v1 are not the same len");

    const double* v0_data = static_cast<double *>(v0_info.ptr);
    const double* v1_data = static_cast<double *>(v1_info.ptr);

    double sum = 0.0;
    for(size_t i=0; i<v0_info.shape[0]; i++){
	sum += v0_data[i]*v1_data[i];
    }
    return sum;
}

py::array_t<double> dgemm_numpy(double alpha,
	           py::array_t<double> A,
		   py::array_t<double> B)
{
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();

    if(A_info.ndim != 2)
	throw std::runtime_error("A is not a vector");
    if(B_info.ndim != 2)
	throw std::runtime_error("B is not a vector");

    if(A_info.shape[1] != B_info.shape[0])
	throw std::runtime_error("Rows of A != Cols of B");


    size_t C_nrows = A_info.shape[0];
    size_t C_ncols = B_info.shape[1];
    size_t n_k = A_info.shape[1]; //same a B_info.shape[0]

    const double* A_data = static_cast<double *>(A_info.ptr);
    const double* B_data = static_cast<double *>(B_info.ptr);

    std::vector<double> C_data(C_nrows*C_ncols);
    
    for(size_t i=0; i< C_nrows; i++){
	for(size_t j=0; j< C_ncols; j++){

	    double val = 0.0;
	    for(size_t k=0; k<n_k; k++){
		val += A_data[i*n_k +k] * B_data[k*C_ncols +j];
	    }
	    C_data[i*C_ncols +j] = alpha*val;

	}
    }

    py::buffer_info Cbuf = 
    {
	C_data.data(),
	sizeof(double),
	py::format_descriptor<double>::format(),
	2,
	{ C_nrows, C_ncols },
	{ C_ncols*sizeof(double), sizeof(double) }
    };

    return py::array_t<double>(Cbuf);
}



py::array_t<double> einJ(py::array_t<double> A,
                         py::array_t<double> B)
{
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();

    if(A_info.ndim != 4)
	throw std::runtime_error("A is not a 4D tensor");
    if(B_info.ndim != 2)
	throw std::runtime_error("B is not a 2D tensor");

    if(A_info.shape[2] != B_info.shape[0])
	throw std::runtime_error("Dimension mismatch A[2] with B[0]");

    if(A_info.shape[3] != B_info.shape[1])
	throw std::runtime_error("Dimension mismatch A[3] with B[1]");

    size_t C_nrows = A_info.shape[0];
    size_t C_ncols = A_info.shape[1];
    size_t n_k = A_info.shape[2]; 
    size_t n_l = A_info.shape[3]; 

    const double* A_data = static_cast<double *>(A_info.ptr);
    const double* B_data = static_cast<double *>(B_info.ptr);

    std::vector<double> C_data(C_nrows*C_ncols);
    
    for(size_t i=0; i< C_nrows; i++){
	for(size_t j=0; j< C_ncols; j++){

	    double val = 0.0;
	    for(size_t k=0; k<n_k; k++){
        for(size_t l=0; l<n_l; l++){
		    val += A_data[l + k * n_l + j * n_k + i * C_ncols] * B_data[l + k * n_l];
        }
	    }
	    C_data[i*C_ncols +j] = val;

	}



//Define interfaces explicitly
PYBIND11_PLUGIN(basic_mod)
{
    py::module m("basic_mod", "QM10 basic module");
    //Fill module here

    //module define(python name, c func name, help str)
    m.def("dot_prod", &dot_prod, "Calculates dot product");
    m.def("dot_prod_np", &dot_prod_np, "Calculates dot product");
    m.def("dgemm_numpy", &dgemm_numpy, "Calculates matrix product");
    m.def("einJ", &einJ, "Computes Einstein summation necessary to construct Coulomb matrix"
//    m.def("einK", &einK, "Computes Einstein summation necessary to construct exchange matrix"


    return m.ptr();
}


