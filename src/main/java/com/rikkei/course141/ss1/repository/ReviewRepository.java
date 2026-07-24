package com.rikkei.course141.ss1.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

import com.rikkei.course141.ss1.model.Review;

public interface ReviewRepository extends JpaRepository<Review, Long> {
    List<Review> findByProductId(Long productId);
}
