"""
Gestational Age Calculator based on Fetal Head Circumference
Uses Hadlock formula for GA estimation
"""
import math


class GestationalAgeCalculator:
    """Calculate gestational age from fetal head circumference"""
    
    @staticmethod
    def calculate_ga_from_hc(hc_mm):
        """
        Calculate gestational age from head circumference using Hadlock formula
        
        Formula: GA (weeks) = exp(1.854 + 0.010451 * HC - 0.000029919 * HC^2 + 0.000000043156 * HC^3)
        
        Args:
            hc_mm: Head circumference in millimeters
            
        Returns:
            dict with:
                - weeks: Full weeks of gestation
                - days: Additional days
                - total_weeks: Decimal weeks (e.g., 24.3)
                - formatted: String like "24 weeks 2 days"
        """
        if hc_mm is None or hc_mm <= 0:
            return None
        
        # Hadlock formula for GA from HC
        # GA in weeks = exp(1.854 + 0.010451*HC - 0.000029919*HC^2 + 0.000000043156*HC^3)
        log_ga = 1.854 + (0.010451 * hc_mm) - (0.000029919 * hc_mm**2) + (0.000000043156 * hc_mm**3)
        total_weeks = math.exp(log_ga)
        
        # Calculate weeks and days
        weeks = int(total_weeks)
        days = int((total_weeks - weeks) * 7)
        
        # Format string
        formatted = f"{weeks} weeks {days} days"
        
        return {
            'weeks': weeks,
            'days': days,
            'total_weeks': round(total_weeks, 1),
            'formatted': formatted
        }
    
    @staticmethod
    def get_percentile_info(hc_mm, ga_weeks):
        """
        Get percentile information for HC at given GA
        (Simplified - in production, use proper growth charts)
        
        Args:
            hc_mm: Head circumference in mm
            ga_weeks: Gestational age in weeks
            
        Returns:
            dict with percentile info
        """
        # Simplified percentile calculation
        # In production, use proper fetal growth charts (Intergrowth-21st, WHO, etc.)
        
        # Approximate HC mean and SD by GA (simplified)
        # These are rough estimates - use proper charts in production
        expected_hc = {
            12: 71, 13: 87, 14: 99, 15: 111, 16: 124, 17: 137, 18: 149,
            19: 162, 20: 175, 21: 187, 22: 199, 23: 212, 24: 224, 25: 236,
            26: 248, 27: 260, 28: 272, 29: 283, 30: 294, 31: 305, 32: 316,
            33: 326, 34: 336, 35: 346, 36: 355, 37: 364, 38: 373, 39: 381,
            40: 389
        }
        
        if ga_weeks not in expected_hc:
            return None
        
        mean_hc = expected_hc[ga_weeks]
        sd = 10  # Simplified SD
        
        z_score = (hc_mm - mean_hc) / sd
        
        # Rough percentile from z-score
        if z_score < -2:
            percentile = "<3rd"
        elif z_score < -1:
            percentile = "3rd-10th"
        elif z_score < 1:
            percentile = "10th-90th (Normal)"
        elif z_score < 2:
            percentile = "90th-97th"
        else:
            percentile = ">97th"
        
        return {
            'percentile': percentile,
            'z_score': round(z_score, 2),
            'expected_hc': mean_hc
        }
