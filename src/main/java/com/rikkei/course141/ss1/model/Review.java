package com.rikkei.course141.ss1.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity @Table(name = "reviews") @Data @NoArgsConstructor @AllArgsConstructor @Builder
public class Review {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Long id;
    @ManyToOne private User user;
    @ManyToOne private Product product;
    private int rating;
    private String comment;
    private LocalDateTime createdDate;
}
