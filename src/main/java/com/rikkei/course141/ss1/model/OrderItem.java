package com.rikkei.course141.ss1.model;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;

@Entity @Table(name = "order_items") @Data @NoArgsConstructor @AllArgsConstructor @Builder
public class OrderItem {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @ManyToOne private Order order;
    @ManyToOne private Product product;
    private int quantity;
    private BigDecimal priceBuy;
}
