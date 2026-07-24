package com.rikkei.course141.ss1.security;

import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import com.rikkei.course141.ss1.model.User;
import com.rikkei.course141.ss1.repository.UserRepository;

@Service
public class UserDetailServiceCustom implements UserDetailsService {
    private final UserRepository userRepository;
    public UserDetailServiceCustom(UserRepository userRepository) { this.userRepository = userRepository; }
    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        User user = userRepository.findByEmail(email).orElseThrow(() -> new UsernameNotFoundException("User not found"));
        return new UserPrincipal(user);
    }
}
