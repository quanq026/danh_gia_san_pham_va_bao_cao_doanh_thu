package com.rikkei.course141.ss1.controller;

import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

import com.rikkei.course141.ss1.dto.request.ReviewRequest;
import com.rikkei.course141.ss1.dto.response.ApiResponse;
import com.rikkei.course141.ss1.model.Product;
import com.rikkei.course141.ss1.model.Review;
import com.rikkei.course141.ss1.model.User;
import com.rikkei.course141.ss1.repository.OrderItemRepository;
import com.rikkei.course141.ss1.repository.OrderRepository;
import com.rikkei.course141.ss1.repository.ProductRepository;
import com.rikkei.course141.ss1.repository.ReviewRepository;
import com.rikkei.course141.ss1.repository.UserRepository;

@RestController
@RequestMapping("/api")
public class ReviewController {
    private final ReviewRepository reviewRepository;
    private final UserRepository userRepository;
    private final ProductRepository productRepository;
    private final OrderRepository orderRepository;
    private final OrderItemRepository orderItemRepository;

    public ReviewController(ReviewRepository reviewRepository, UserRepository userRepository,
                            ProductRepository productRepository, OrderRepository orderRepository,
                            OrderItemRepository orderItemRepository) {
        this.reviewRepository = reviewRepository;
        this.userRepository = userRepository;
        this.productRepository = productRepository;
        this.orderRepository = orderRepository;
        this.orderItemRepository = orderItemRepository;
    }

    @PostMapping("/reviews") public ResponseEntity<ApiResponse<Review>> create(@Valid @RequestBody ReviewRequest dto, Authentication auth) {
        User user = userRepository.findByEmail(auth.getName()).orElseThrow();
        Product product = productRepository.findById(dto.getProductId()).orElseThrow();
        // Check bought
        boolean bought = orderRepository.findByUserEmail(user.getEmail()).stream()
            .flatMap(o -> o.getItems().stream())
            .anyMatch(i -> i.getProduct().getId().equals(product.getId()));
        if (!bought) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(ApiResponse.error(403, "Bạn chưa mua sản phẩm này"));
        }
        Review review = Review.builder().user(user).product(product).rating(dto.getRating())
            .comment(dto.getComment()).createdDate(LocalDateTime.now()).build();
        return ResponseEntity.status(HttpStatus.CREATED).body(ApiResponse.success(reviewRepository.save(review)));
    }

    @GetMapping("/products/{id}/reviews") public ResponseEntity<ApiResponse<List<Review>>> getByProduct(@PathVariable Long id) {
        return ResponseEntity.ok(ApiResponse.success(reviewRepository.findByProductId(id)));
    }
}
