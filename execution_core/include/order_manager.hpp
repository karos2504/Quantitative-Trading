#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace hft {

// Packed struct for minimal memory footprint in Kernel Bypass DMA buffers
#pragma pack(push, 1)
struct Order {
    uint64_t timestamp_ns;
    char symbol[8];
    double price;
    int32_t quantity;
    char side; // 'B' or 'S'
    char order_type; // 'L' or 'M'
};
#pragma pack(pop)

class OrderManager {
public:
    OrderManager(const std::string& exchange_ip, int port);
    ~OrderManager();

    // Sends order natively via TCP/UDP bypass mapped immediately to the NIC
    bool submit_order(const std::string& symbol, double price, int qty, char side, char order_type);
    
    // CPU-pinned lock-free polling from ring buffer
    std::vector<Order> poll_completions();

private:
    std::string exchange_ip_;
    int port_;
    // Placeholder for Solarflare EF_VI / DPDK rings
};

} // namespace hft
