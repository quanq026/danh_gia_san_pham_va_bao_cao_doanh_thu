package com.rikkei.course141.ss1.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.rikkei.course141.ss1.dto.response.ApiResponse;
import com.rikkei.course141.ss1.model.Order;
import com.rikkei.course141.ss1.repository.OrderRepository;

@RestController
@RequestMapping("/api/reports")
public class ReportController {
    private final OrderRepository orderRepository;
    public ReportController(OrderRepository orderRepository) { this.orderRepository = orderRepository; }

    @GetMapping("/revenue") public ResponseEntity<ApiResponse<Map<String, BigDecimal>>> revenue(@RequestParam(defaultValue = "month") String type) {
        LocalDateTime now = LocalDateTime.now();
        List<Order> orders = orderRepository.findAll().stream()
            .filter(o -> "COMPLETED".equals(o.getStatus())).collect(Collectors.toList());
        Map<String, BigDecimal> result = new HashMap<>();
        for (Order o : orders) {
            String key = switch (type) {
                case "day" -> o.getCreatedDate().toLocalDate().toString();
                case "year" -> String.valueOf(o.getCreatedDate().getYear());
                default -> o.getCreatedDate().getMonth().toString();
            };
            result.merge(key, o.getTotalMoney(), BigDecimal::add);
        }
        return ResponseEntity.ok(ApiResponse.success(result));
    }
}
