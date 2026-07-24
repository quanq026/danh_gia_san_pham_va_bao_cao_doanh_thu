package com.rikkei.course141.ss1.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

import com.rikkei.course141.ss1.model.User;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
}
