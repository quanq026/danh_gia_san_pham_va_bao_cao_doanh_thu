package com.rikkei.course141.ss1.dto.request;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ReviewRequest {
    @NotNull private Long productId;
    @Min(1) @Max(5) private int rating;
    private String comment;
}
