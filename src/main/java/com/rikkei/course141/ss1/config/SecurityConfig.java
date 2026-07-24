package com.rikkei.course141.ss1.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.ProviderManager;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import com.rikkei.course141.ss1.security.UserDetailServiceCustom;

@Configuration
public class SecurityConfig {
    private final UserDetailServiceCustom userDetailsService;
    private final JwtAuthenticationFilter jwtFilter;
    public SecurityConfig(UserDetailServiceCustom userDetailsService, JwtAuthenticationFilter jwtFilter) {
        this.userDetailsService = userDetailsService;
        this.jwtFilter = jwtFilter;
    }
    @Bean public PasswordEncoder passwordEncoder() { return new BCryptPasswordEncoder(); }
    @Bean public AuthenticationManager authenticationManager() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder());
        return new ProviderManager(provider);
    }
    @Bean public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.csrf(csrf -> csrf.disable())
            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/products").permitAll()
                .requestMatchers("/api/products/**").hasAnyRole("ADMIN", "STAFF")
                .requestMatchers("/api/orders", "/api/orders/my").hasRole("CUSTOMER")
                .requestMatchers("/api/orders/**").hasAnyRole("ADMIN", "STAFF")
                .requestMatchers("/api/users/me").authenticated()
                .requestMatchers("/api/users/**").hasRole("ADMIN")
                .requestMatchers("/api/reviews").hasRole("CUSTOMER")
                .requestMatchers("/api/reports/**").hasRole("ADMIN")
                .anyRequest().authenticated())
            .addFilterBefore(jwtFilter, UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }
}
