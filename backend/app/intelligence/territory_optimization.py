"""
Territory Optimization Engine
AI-powered territory and quota planning with geographic analysis and performance optimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TerritoryType(Enum):
    """Types of sales territories"""
    GEOGRAPHIC = "geographic"
    INDUSTRY_VERTICAL = "industry_vertical"
    ACCOUNT_SIZE = "account_size"
    PRODUCT_LINE = "product_line"
    HYBRID = "hybrid"

class OptimizationObjective(Enum):
    """Territory optimization objectives"""
    REVENUE_MAXIMIZATION = "revenue_maximization"
    COST_MINIMIZATION = "cost_minimization"
    BALANCED_WORKLOAD = "balanced_workload"
    TRAVEL_EFFICIENCY = "travel_efficiency"
    COVERAGE_MAXIMIZATION = "coverage_maximization"
    CUSTOMER_SATISFACTION = "customer_satisfaction"

class TerritoryStatus(Enum):
    """Territory status"""
    ACTIVE = "active"
    PROPOSED = "proposed"
    INACTIVE = "inactive"
    UNDER_REVIEW = "under_review"

class RepType(Enum):
    """Sales representative types"""
    INSIDE_SALES = "inside_sales"
    FIELD_SALES = "field_sales"
    ENTERPRISE = "enterprise"
    SMB = "smb"
    CHANNEL_PARTNER = "channel_partner"
    TECHNICAL_SALES = "technical_sales"

@dataclass
class GeographicCoordinate:
    """Geographic coordinate with metadata"""
    latitude: float
    longitude: float
    address: str
    city: str
    state: str
    country: str
    postal_code: str

@dataclass
class Customer:
    """Customer entity for territory planning"""
    customer_id: str
    name: str
    coordinates: GeographicCoordinate
    annual_revenue: float
    industry: str
    size_category: str  # SMB, Mid-market, Enterprise
    last_purchase_date: datetime
    lifetime_value: float
    growth_potential: float  # 0-1 scale
    current_rep_id: Optional[str]
    satisfaction_score: float
    contact_frequency: int  # visits per year
    decision_complexity: int  # 1-5 scale

@dataclass
class SalesRep:
    """Sales representative for territory assignment"""
    rep_id: str
    name: str
    rep_type: RepType
    base_location: GeographicCoordinate
    max_travel_distance_km: float
    capacity_score: float  # 0-1 scale
    performance_rating: float  # 0-10 scale
    industry_expertise: List[str]
    language_skills: List[str]
    current_territory_id: Optional[str]
    quota_target: float
    current_pipeline: float
    win_rate: float

@dataclass
class Territory:
    """Sales territory definition"""
    territory_id: str
    name: str
    territory_type: TerritoryType
    assigned_rep_id: Optional[str]
    customers: List[str]  # Customer IDs
    boundaries: Dict[str, Any]  # Geographic or criteria boundaries
    annual_quota: float
    estimated_revenue: float
    travel_budget: float
    status: TerritoryStatus
    created_date: datetime
    last_optimized: datetime
    performance_metrics: Dict[str, float]
    optimization_score: float

@dataclass
class TerritoryOptimization:
    """Territory optimization result"""
    optimization_id: str
    objective: OptimizationObjective
    original_territories: List[Territory]
    optimized_territories: List[Territory]
    performance_improvement: Dict[str, float]
    rep_assignments: Dict[str, str]  # rep_id -> territory_id
    optimization_date: datetime
    confidence_score: float
    implementation_plan: List[Dict[str, Any]]

@dataclass
class QuotaRecommendation:
    """Quota planning recommendation"""
    recommendation_id: str
    rep_id: str
    territory_id: str
    current_quota: float
    recommended_quota: float
    quota_change_percentage: float
    justification: str
    historical_performance: Dict[str, float]
    market_factors: List[str]
    risk_assessment: str
    confidence_level: float

class TerritoryOptimizationEngine:
    """Advanced AI-powered territory optimization system"""
    
    def __init__(self, config_path: str = "config/territory_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Data storage
        self.customers: Dict[str, Customer] = {}
        self.sales_reps: Dict[str, SalesRep] = {}
        self.territories: Dict[str, Territory] = {}
        self.optimizations: Dict[str, TerritoryOptimization] = {}
        self.quota_recommendations: Dict[str, QuotaRecommendation] = {}
        
        # ML models and utilities
        self.scaler = StandardScaler()
        self.geocoder = Nominatim(user_agent="territory_optimizer")
        
        # Optimization algorithms
        self.clustering_algorithm = KMeans(n_clusters=10, random_state=42)
        
        logger.info("TerritoryOptimizationEngine initialized")
    
    async def optimize_territories(
        self,
        optimization_objective: OptimizationObjective = OptimizationObjective.REVENUE_MAXIMIZATION,
        constraints: Dict[str, Any] = None,
        rebalance_existing: bool = True
    ) -> TerritoryOptimization:
        """Optimize territory assignments using AI algorithms"""
        try:
            if constraints is None:
                constraints = self._get_default_constraints()
            
            # Prepare optimization data
            customer_data = await self._prepare_customer_data()
            rep_data = await self._prepare_rep_data()
            
            if not customer_data or not rep_data:
                raise ValueError("Insufficient data for territory optimization")
            
            # Current state analysis
            current_territories = list(self.territories.values())
            current_performance = await self._analyze_current_performance()
            
            # Run optimization algorithm
            optimized_assignments = await self._run_optimization_algorithm(
                customer_data, rep_data, optimization_objective, constraints
            )
            
            # Generate new territory definitions
            optimized_territories = await self._generate_optimized_territories(
                optimized_assignments, optimization_objective
            )
            
            # Calculate performance improvement
            performance_improvement = await self._calculate_performance_improvement(
                current_territories, optimized_territories
            )
            
            # Generate implementation plan
            implementation_plan = await self._generate_implementation_plan(
                current_territories, optimized_territories
            )
            
            # Create optimization result
            optimization = TerritoryOptimization(
                optimization_id=f"opt_{int(datetime.now().timestamp())}",
                objective=optimization_objective,
                original_territories=current_territories,
                optimized_territories=optimized_territories,
                performance_improvement=performance_improvement,
                rep_assignments=optimized_assignments["rep_assignments"],
                optimization_date=datetime.now(),
                confidence_score=optimized_assignments["confidence_score"],
                implementation_plan=implementation_plan
            )
            
            # Store optimization
            self.optimizations[optimization.optimization_id] = optimization
            
            logger.info(f"Territory optimization completed with {performance_improvement.get('revenue_improvement', 0):.2f}% revenue improvement")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing territories: {e}")
            raise
    
    async def optimize_quotas(
        self,
        territory_ids: List[str] = None,
        quota_period: str = "annual",
        market_growth_factor: float = 1.05
    ) -> Dict[str, QuotaRecommendation]:
        """Generate AI-powered quota recommendations"""
        try:
            if territory_ids is None:
                territory_ids = list(self.territories.keys())
            
            quota_recommendations = {}
            
            for territory_id in territory_ids:
                territory = self.territories.get(territory_id)
                if not territory or not territory.assigned_rep_id:
                    continue
                
                rep = self.sales_reps.get(territory.assigned_rep_id)
                if not rep:
                    continue
                
                # Analyze historical performance
                historical_data = await self._get_historical_performance(territory_id)
                
                # Calculate market potential
                market_potential = await self._calculate_market_potential(territory)
                
                # Factor in rep performance and capacity
                rep_factor = await self._calculate_rep_factor(rep)
                
                # Generate quota recommendation
                recommendation = await self._calculate_quota_recommendation(
                    territory, rep, historical_data, market_potential, 
                    rep_factor, market_growth_factor
                )
                
                if recommendation:
                    quota_recommendations[territory_id] = recommendation
                    self.quota_recommendations[recommendation.recommendation_id] = recommendation
            
            logger.info(f"Generated quota recommendations for {len(quota_recommendations)} territories")
            return quota_recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing quotas: {e}")
            return {}
    
    async def analyze_territory_coverage(
        self,
        territory_ids: List[str] = None,
        include_gaps: bool = True
    ) -> Dict[str, Any]:
        """Analyze territory coverage and identify gaps"""
        try:
            if territory_ids is None:
                territory_ids = list(self.territories.keys())
            
            coverage_analysis = {
                "analysis_date": datetime.now().isoformat(),
                "territories_analyzed": len(territory_ids),
                "overall_coverage": {},
                "territory_details": {},
                "coverage_gaps": [],
                "overlaps": [],
                "recommendations": []
            }
            
            # Analyze each territory
            total_customers = len(self.customers)
            covered_customers = set()
            territory_coverage = {}
            
            for territory_id in territory_ids:
                territory = self.territories.get(territory_id)
                if not territory:
                    continue
                
                # Calculate territory coverage metrics
                territory_metrics = await self._calculate_territory_metrics(territory)
                territory_coverage[territory_id] = territory_metrics
                
                # Track covered customers
                covered_customers.update(territory.customers)
                
                coverage_analysis["territory_details"][territory_id] = {
                    "customer_count": len(territory.customers),
                    "revenue_potential": territory.estimated_revenue,
                    "geographic_area": territory_metrics.get("geographic_area_km2", 0),
                    "customer_density": territory_metrics.get("customer_density", 0),
                    "travel_efficiency": territory_metrics.get("travel_efficiency", 0),
                    "rep_utilization": territory_metrics.get("rep_utilization", 0)
                }
            
            # Overall coverage metrics
            coverage_analysis["overall_coverage"] = {
                "customer_coverage_percentage": (len(covered_customers) / total_customers * 100) if total_customers > 0 else 0,
                "uncovered_customers": total_customers - len(covered_customers),
                "average_territory_size": np.mean([len(t.customers) for t in self.territories.values()]),
                "territory_balance_score": await self._calculate_territory_balance_score()
            }
            
            # Identify coverage gaps
            if include_gaps:
                coverage_analysis["coverage_gaps"] = await self._identify_coverage_gaps(covered_customers)
                coverage_analysis["overlaps"] = await self._identify_territory_overlaps()
            
            # Generate recommendations
            coverage_analysis["recommendations"] = await self._generate_coverage_recommendations(
                coverage_analysis["overall_coverage"],
                coverage_analysis["coverage_gaps"],
                coverage_analysis["overlaps"]
            )
            
            return coverage_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing territory coverage: {e}")
            return {"error": str(e)}
    
    async def recommend_territory_adjustments(
        self,
        performance_threshold: float = 0.8,
        balance_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Recommend territory adjustments based on performance and balance"""
        try:
            recommendations = []
            
            # Analyze each territory performance
            for territory_id, territory in self.territories.items():
                if territory.status != TerritoryStatus.ACTIVE:
                    continue
                
                # Calculate performance metrics
                performance_score = await self._calculate_territory_performance_score(territory)
                balance_score = await self._calculate_territory_balance_score_single(territory)
                
                # Check if adjustments are needed
                needs_performance_adjustment = performance_score < performance_threshold
                needs_balance_adjustment = abs(balance_score - 0.5) > balance_threshold
                
                if needs_performance_adjustment or needs_balance_adjustment:
                    adjustment = await self._generate_territory_adjustment(
                        territory, performance_score, balance_score
                    )
                    if adjustment:
                        recommendations.append(adjustment)
            
            # Sort recommendations by priority
            recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} territory adjustment recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating territory adjustment recommendations: {e}")
            return []
    
    async def simulate_territory_changes(
        self,
        proposed_changes: List[Dict[str, Any]],
        simulation_period_months: int = 12
    ) -> Dict[str, Any]:
        """Simulate the impact of proposed territory changes"""
        try:
            simulation_results = {
                "simulation_id": f"sim_{int(datetime.now().timestamp())}",
                "simulation_date": datetime.now().isoformat(),
                "period_months": simulation_period_months,
                "proposed_changes": proposed_changes,
                "baseline_metrics": {},
                "projected_metrics": {},
                "impact_analysis": {},
                "risk_assessment": {},
                "recommendations": []
            }
            
            # Calculate baseline metrics
            baseline_metrics = await self._calculate_baseline_metrics()
            simulation_results["baseline_metrics"] = baseline_metrics
            
            # Apply proposed changes and calculate projected metrics
            projected_metrics = await self._simulate_changed_metrics(
                proposed_changes, baseline_metrics, simulation_period_months
            )
            simulation_results["projected_metrics"] = projected_metrics
            
            # Analyze impact
            impact_analysis = await self._analyze_simulation_impact(
                baseline_metrics, projected_metrics
            )
            simulation_results["impact_analysis"] = impact_analysis
            
            # Assess risks
            risk_assessment = await self._assess_change_risks(proposed_changes, impact_analysis)
            simulation_results["risk_assessment"] = risk_assessment
            
            # Generate recommendations
            recommendations = await self._generate_simulation_recommendations(
                impact_analysis, risk_assessment
            )
            simulation_results["recommendations"] = recommendations
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error simulating territory changes: {e}")
            return {"error": str(e)}
    
    async def get_territory_analytics_dashboard(
        self,
        time_period_days: int = 90,
        include_predictive: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive territory analytics dashboard"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            dashboard = {
                "generated_at": datetime.now().isoformat(),
                "period_days": time_period_days,
                "territory_overview": {},
                "performance_metrics": {},
                "optimization_opportunities": {},
                "rep_utilization": {},
                "geographic_insights": {},
                "quota_analysis": {},
                "trend_analysis": {}
            }
            
            # Territory overview
            dashboard["territory_overview"] = await self._get_territory_overview()
            
            # Performance metrics
            dashboard["performance_metrics"] = await self._get_performance_metrics(cutoff_date)
            
            # Optimization opportunities
            dashboard["optimization_opportunities"] = await self._identify_optimization_opportunities()
            
            # Rep utilization analysis
            dashboard["rep_utilization"] = await self._analyze_rep_utilization()
            
            # Geographic insights
            dashboard["geographic_insights"] = await self._generate_geographic_insights()
            
            # Quota analysis
            dashboard["quota_analysis"] = await self._analyze_quota_performance()
            
            # Trend analysis
            if include_predictive:
                dashboard["trend_analysis"] = await self._analyze_territory_trends(cutoff_date)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating territory analytics dashboard: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _prepare_customer_data(self) -> List[Dict[str, Any]]:
        """Prepare customer data for optimization algorithms"""
        customer_data = []
        
        for customer in self.customers.values():
            data = {
                "customer_id": customer.customer_id,
                "latitude": customer.coordinates.latitude,
                "longitude": customer.coordinates.longitude,
                "annual_revenue": customer.annual_revenue,
                "lifetime_value": customer.lifetime_value,
                "growth_potential": customer.growth_potential,
                "satisfaction_score": customer.satisfaction_score,
                "contact_frequency": customer.contact_frequency,
                "decision_complexity": customer.decision_complexity
            }
            customer_data.append(data)
        
        return customer_data
    
    async def _prepare_rep_data(self) -> List[Dict[str, Any]]:
        """Prepare sales rep data for optimization"""
        rep_data = []
        
        for rep in self.sales_reps.values():
            data = {
                "rep_id": rep.rep_id,
                "latitude": rep.base_location.latitude,
                "longitude": rep.base_location.longitude,
                "max_travel_distance": rep.max_travel_distance_km,
                "capacity_score": rep.capacity_score,
                "performance_rating": rep.performance_rating,
                "quota_target": rep.quota_target,
                "win_rate": rep.win_rate
            }
            rep_data.append(data)
        
        return rep_data
    
    async def _run_optimization_algorithm(
        self,
        customer_data: List[Dict[str, Any]],
        rep_data: List[Dict[str, Any]],
        objective: OptimizationObjective,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the core optimization algorithm"""
        
        # Convert to matrices for optimization
        customer_matrix = np.array([
            [c["latitude"], c["longitude"], c["annual_revenue"], 
             c["growth_potential"], c["satisfaction_score"]]
            for c in customer_data
        ])
        
        rep_matrix = np.array([
            [r["latitude"], r["longitude"], r["capacity_score"], 
             r["performance_rating"], r["quota_target"]]
            for r in rep_data
        ])
        
        # Normalize features
        customer_scaled = self.scaler.fit_transform(customer_matrix)
        
        # Apply clustering algorithm based on objective
        if objective == OptimizationObjective.REVENUE_MAXIMIZATION:
            assignments = await self._optimize_for_revenue(customer_scaled, rep_matrix)
        elif objective == OptimizationObjective.TRAVEL_EFFICIENCY:
            assignments = await self._optimize_for_travel(customer_scaled, rep_matrix)
        elif objective == OptimizationObjective.BALANCED_WORKLOAD:
            assignments = await self._optimize_for_balance(customer_scaled, rep_matrix)
        else:
            # Default to balanced approach
            assignments = await self._optimize_balanced_approach(customer_scaled, rep_matrix)
        
        return {
            "rep_assignments": assignments,
            "confidence_score": 0.85,  # Would be calculated based on algorithm performance
            "optimization_metrics": {
                "total_revenue_potential": sum(c["annual_revenue"] for c in customer_data),
                "average_travel_distance": np.mean([c.get("travel_distance", 0) for c in customer_data]),
                "workload_balance_score": 0.78
            }
        }
    
    async def _optimize_for_revenue(
        self, customer_matrix: np.ndarray, rep_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Optimize territory assignments for maximum revenue"""
        
        assignments = {}
        
        # Simple assignment based on rep performance and customer value
        for i, customer in enumerate(customer_matrix):
            best_rep_idx = 0
            best_score = 0
            
            for j, rep in enumerate(rep_matrix):
                # Calculate distance
                distance = np.sqrt((customer[0] - rep[0])**2 + (customer[1] - rep[1])**2)
                
                # Calculate assignment score (higher is better)
                if distance < 0.1:  # Within reasonable distance (normalized)
                    score = rep[3] * rep[2]  # performance * capacity
                    if score > best_score:
                        best_score = score
                        best_rep_idx = j
            
            customer_id = list(self.customers.keys())[i]
            rep_id = list(self.sales_reps.keys())[best_rep_idx]
            assignments[customer_id] = rep_id
        
        return assignments
    
    async def _optimize_for_travel(
        self, customer_matrix: np.ndarray, rep_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Optimize territory assignments for travel efficiency"""
        
        assignments = {}
        
        # Assign customers to nearest qualified reps
        for i, customer in enumerate(customer_matrix):
            min_distance = float('inf')
            best_rep_idx = 0
            
            for j, rep in enumerate(rep_matrix):
                distance = np.sqrt((customer[0] - rep[0])**2 + (customer[1] - rep[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_rep_idx = j
            
            customer_id = list(self.customers.keys())[i]
            rep_id = list(self.sales_reps.keys())[best_rep_idx]
            assignments[customer_id] = rep_id
        
        return assignments
    
    async def _optimize_for_balance(
        self, customer_matrix: np.ndarray, rep_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Optimize for balanced workload distribution"""
        
        assignments = {}
        rep_workloads = {rep_id: 0 for rep_id in self.sales_reps.keys()}
        
        # Sort customers by value (descending)
        customer_indices = sorted(
            range(len(customer_matrix)),
            key=lambda i: customer_matrix[i][2],  # Annual revenue
            reverse=True
        )
        
        for i in customer_indices:
            customer = customer_matrix[i]
            
            # Find rep with lowest current workload who can handle this customer
            best_rep_idx = min(
                range(len(rep_matrix)),
                key=lambda j: rep_workloads[list(self.sales_reps.keys())[j]]
            )
            
            customer_id = list(self.customers.keys())[i]
            rep_id = list(self.sales_reps.keys())[best_rep_idx]
            assignments[customer_id] = rep_id
            
            # Update workload
            rep_workloads[rep_id] += customer[2]  # Add revenue to workload
        
        return assignments
    
    async def _optimize_balanced_approach(
        self, customer_matrix: np.ndarray, rep_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Balanced optimization considering multiple factors"""
        
        assignments = {}
        
        for i, customer in enumerate(customer_matrix):
            best_rep_idx = 0
            best_score = -1
            
            for j, rep in enumerate(rep_matrix):
                # Calculate multi-factor score
                distance = np.sqrt((customer[0] - rep[0])**2 + (customer[1] - rep[1])**2)
                distance_score = max(0, 1 - distance * 10)  # Closer is better
                
                performance_score = rep[3] / 10  # Normalize performance rating
                capacity_score = rep[2]  # Capacity score
                
                # Combined score with weights
                combined_score = (
                    distance_score * 0.4 +
                    performance_score * 0.4 +
                    capacity_score * 0.2
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_rep_idx = j
            
            customer_id = list(self.customers.keys())[i]
            rep_id = list(self.sales_reps.keys())[best_rep_idx]
            assignments[customer_id] = rep_id
        
        return assignments
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """Get default optimization constraints"""
        return {
            "max_customers_per_rep": 50,
            "min_customers_per_rep": 5,
            "max_travel_distance_km": 500,
            "min_quota_per_territory": 100000,
            "max_quota_per_territory": 2000000,
            "balance_tolerance": 0.2
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load territory optimization configuration"""
        return {
            "optimization_frequency_days": 30,
            "performance_weights": {
                "revenue": 0.4,
                "efficiency": 0.3,
                "balance": 0.3
            },
            "default_travel_budget": 15000,
            "quota_adjustment_limit": 0.25  # Max 25% quota change
        }


# Initialize sample data for demonstration
def initialize_sample_data(engine: TerritoryOptimizationEngine):
    """Initialize sample customers and reps for demonstration"""
    
    # Sample customers
    customers = [
        Customer(
            customer_id="cust_001",
            name="TechCorp Inc",
            coordinates=GeographicCoordinate(37.7749, -122.4194, "123 Market St", "San Francisco", "CA", "USA", "94102"),
            annual_revenue=500000,
            industry="Technology",
            size_category="Mid-market",
            last_purchase_date=datetime(2024, 6, 15),
            lifetime_value=750000,
            growth_potential=0.8,
            current_rep_id=None,
            satisfaction_score=8.5,
            contact_frequency=12,
            decision_complexity=3
        ),
        Customer(
            customer_id="cust_002", 
            name="Global Manufacturing",
            coordinates=GeographicCoordinate(34.0522, -118.2437, "456 Industry Blvd", "Los Angeles", "CA", "USA", "90210"),
            annual_revenue=1200000,
            industry="Manufacturing",
            size_category="Enterprise",
            last_purchase_date=datetime(2024, 8, 20),
            lifetime_value=2000000,
            growth_potential=0.6,
            current_rep_id=None,
            satisfaction_score=9.2,
            contact_frequency=18,
            decision_complexity=5
        )
    ]
    
    # Sample sales reps
    reps = [
        SalesRep(
            rep_id="rep_001",
            name="Sarah Johnson",
            rep_type=RepType.FIELD_SALES,
            base_location=GeographicCoordinate(37.7849, -122.4094, "789 Sales St", "San Francisco", "CA", "USA", "94103"),
            max_travel_distance_km=300,
            capacity_score=0.85,
            performance_rating=8.7,
            industry_expertise=["Technology", "Software"],
            language_skills=["English", "Spanish"],
            current_territory_id=None,
            quota_target=800000,
            current_pipeline=650000,
            win_rate=0.42
        ),
        SalesRep(
            rep_id="rep_002",
            name="Mike Chen",
            rep_type=RepType.ENTERPRISE,
            base_location=GeographicCoordinate(34.0622, -118.2537, "321 Enterprise Way", "Los Angeles", "CA", "USA", "90211"),
            max_travel_distance_km=400,
            capacity_score=0.92,
            performance_rating=9.1,
            industry_expertise=["Manufacturing", "Industrial"],
            language_skills=["English", "Mandarin"],
            current_territory_id=None,
            quota_target=1500000,
            current_pipeline=1200000,
            win_rate=0.38
        )
    ]
    
    # Add to engine
    for customer in customers:
        engine.customers[customer.customer_id] = customer
    
    for rep in reps:
        engine.sales_reps[rep.rep_id] = rep


# Global instance
territory_optimizer = TerritoryOptimizationEngine()
initialize_sample_data(territory_optimizer)