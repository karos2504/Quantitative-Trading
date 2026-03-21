#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "../include/order_manager.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hft_execution, m) {
    m.doc() = "C++ High-Frequency Order Manager Interface mapped to Python Event Engine";

    py::class_<hft::OrderManager>(m, "OrderManager")
        .def(py::init<std::string, int>(), py::arg("exchange_ip"), py::arg("port"))
        .def("submit_order", &hft::OrderManager::submit_order, 
             "Directly push an order into the hardware ring-buffer.",
             py::arg("symbol"), py::arg("price"), py::arg("qty"), py::arg("side"), py::arg("order_type"))
        .def("poll_completions", &hft::OrderManager::poll_completions,
             "Consume lock-free struct memory from network card.");
}
