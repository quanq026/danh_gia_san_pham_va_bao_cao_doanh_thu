package com.rikkei.course141.ss1.config;

import io.jsonwebtoken.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import java.util.Date;

@Component
public class JwtProvider {
    @Value("${app.jwt.secret}")
    private String secret;
    @Value("${app.jwt.expiration}")
    private long expiration;
    @Value("${app.jwt.refresh-expiration:604800000}")
    private long refreshExpiration;

    public String generateToken(String email, String role) {
        Date now = new Date();
        return Jwts.builder().setSubject(email).claim("role", role).setIssuedAt(now)
            .setExpiration(new Date(now.getTime() + expiration))
            .signWith(SignatureAlgorithm.HS512, secret).compact();
    }

    public String generateRefreshToken(String email) {
        Date now = new Date();
        return Jwts.builder().setSubject(email).setIssuedAt(now)
            .setExpiration(new Date(now.getTime() + refreshExpiration))
            .signWith(SignatureAlgorithm.HS512, secret).compact();
    }

    public String getEmail(String token) {
        return Jwts.parser().setSigningKey(secret).parseClaimsJws(token).getBody().getSubject();
    }

    public boolean validateToken(String token) {
        try { Jwts.parser().setSigningKey(secret).parseClaimsJws(token); return true; }
        catch (Exception e) { return false; }
    }
}
