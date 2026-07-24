package com.rikkei.course141.ss1.model;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Entity @Table(name = "orders") @Data @NoArgsConstructor @AllArgsConstructor @Builder
public class Order {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @ManyToOne private User user;
    private LocalDateTime createdDate;
    private String status;
    private BigDecimal totalMoney;
    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL) private List<OrderItem> items;
}
