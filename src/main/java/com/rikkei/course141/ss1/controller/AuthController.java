package com.rikkei.course141.ss1.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

import com.rikkei.course141.ss1.config.JwtProvider;
import com.rikkei.course141.ss1.dto.response.ApiResponse;
import com.rikkei.course141.ss1.model.User;
import com.rikkei.course141.ss1.repository.UserRepository;
import com.rikkei.course141.ss1.security.UserPrincipal;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final AuthenticationManager authenticationManager;
    private final JwtProvider jwtProvider;

    public AuthController(UserRepository userRepository, PasswordEncoder passwordEncoder,
                          AuthenticationManager authenticationManager, JwtProvider jwtProvider) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.authenticationManager = authenticationManager;
        this.jwtProvider = jwtProvider;
    }

    @PostMapping("/register")
    public ResponseEntity<ApiResponse<User>> register(@RequestBody Map<String, String> body) {
        if (userRepository.findByEmail(body.get("email")).isPresent()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(ApiResponse.error(400, "Email đã tồn tại"));
        }
        User user = User.builder().fullName(body.get("fullName")).phone(body.get("phone"))
            .email(body.get("email")).password(passwordEncoder.encode(body.get("password")))
            .role("ROLE_USER").build();
        return ResponseEntity.status(HttpStatus.CREATED).body(ApiResponse.success(userRepository.save(user)));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody Map<String, String> body) {
        Authentication auth = authenticationManager.authenticate(
            new UsernamePasswordAuthenticationToken(body.get("email"), body.get("password")));
        UserPrincipal principal = (UserPrincipal) auth.getPrincipal();
        String accessToken = jwtProvider.generateToken(principal.getUsername(), principal.getUser().getRole());
        String refreshToken = jwtProvider.generateRefreshToken(principal.getUsername());
        return ResponseEntity.ok(Map.of("accessToken", accessToken, "refreshToken", refreshToken, "type", "Bearer", "email", principal.getUsername()));
    }
}
