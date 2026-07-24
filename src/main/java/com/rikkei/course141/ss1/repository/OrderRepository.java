package com.rikkei.course141.ss1.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

import com.rikkei.course141.ss1.model.Order;

public interface OrderRepository extends JpaRepository<Order, Long> {
    List<Order> findByUserEmail(String email);
}
