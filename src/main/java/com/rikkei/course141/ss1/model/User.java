package com.rikkei.course141.ss1.model;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(unique = true)
    private String email;
    private String password;
    private String fullName;
    private String phone;
    private String role; // ROLE_USER, ROLE_ADMIN, ROLE_STAFF, ROLE_CUSTOMER
    private boolean enabled = true;
}
