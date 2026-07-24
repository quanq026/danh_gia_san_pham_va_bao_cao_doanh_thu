package com.rikkei.course141.ss1.config;

import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import com.rikkei.course141.ss1.model.User;
import com.rikkei.course141.ss1.repository.UserRepository;

@Component
public class DataInitializer implements CommandLineRunner {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public DataInitializer(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Override
    public void run(String... args) {
        if (userRepository.count() == 0) {
            userRepository.save(User.builder()
                .email("admin@test.com")
                .password(passwordEncoder.encode("123456"))
                .fullName("Admin")
                .phone("0123456789")
                .role("ROLE_ADMIN")
                .enabled(true)
                .build());
            userRepository.save(User.builder()
                .email("customer@test.com")
                .password(passwordEncoder.encode("123456"))
                .fullName("Customer")
                .phone("0987654321")
                .role("ROLE_CUSTOMER")
                .enabled(true)
                .build());
            userRepository.save(User.builder()
                .email("staff@test.com")
                .password(passwordEncoder.encode("123456"))
                .fullName("Staff")
                .phone("0555888999")
                .role("ROLE_STAFF")
                .enabled(true)
                .build());
            System.out.println("Seed data: admin@test.com/123456, customer@test.com/123456, staff@test.com/123456");
        }
    }
}
