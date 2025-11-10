"""
Skill Gap Analysis Framework
Comprehensive analysis to identify specific areas for sales rep improvement with actionable insights
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class SalesSkill(Enum):
    """Types of sales skills to analyze"""
    PROSPECTING = "prospecting"
    COLD_CALLING = "cold_calling"
    RAPPORT_BUILDING = "rapport_building"
    NEEDS_DISCOVERY = "needs_discovery"
    QUESTIONING_TECHNIQUE = "questioning_technique"
    ACTIVE_LISTENING = "active_listening"
    PRODUCT_KNOWLEDGE = "product_knowledge"
    VALUE_PROPOSITION = "value_proposition"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING_TECHNIQUES = "closing_techniques"
    FOLLOW_UP = "follow_up"
    NEGOTIATION = "negotiation"
    PRESENTATION_SKILLS = "presentation_skills"
    TIME_MANAGEMENT = "time_management"
    PIPELINE_MANAGEMENT = "pipeline_management"
    CUSTOMER_RELATIONSHIP = "customer_relationship"
    COMMUNICATION_CLARITY = "communication_clarity"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"

class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = "novice"           # 0-25
    DEVELOPING = "developing"   # 26-50
    COMPETENT = "competent"     # 51-75
    PROFICIENT = "proficient"   # 76-90
    EXPERT = "expert"           # 91-100

class GapSeverity(Enum):
    """Severity of skill gaps"""
    CRITICAL = "critical"       # >40 point gap
    HIGH = "high"              # 25-40 point gap
    MEDIUM = "medium"          # 15-25 point gap
    LOW = "low"                # 5-15 point gap
    MINIMAL = "minimal"        # <5 point gap

class AnalysisMethod(Enum):
    """Methods for skill assessment"""
    CALL_ANALYSIS = "call_analysis"
    PERFORMANCE_METRICS = "performance_metrics"
    PEER_COMPARISON = "peer_comparison"
    MANAGER_ASSESSMENT = "manager_assessment"
    CUSTOMER_FEEDBACK = "customer_feedback"
    SELF_ASSESSMENT = "self_assessment"

@dataclass
class SkillAssessment:
    """Individual skill assessment"""
    skill: SalesSkill
    current_level: float  # 0-100 score
    skill_level_category: SkillLevel
    assessment_method: AnalysisMethod
    confidence: float  # Confidence in assessment (0-1)
    evidence: List[str]  # Supporting evidence
    assessment_date: datetime
    assessor: str  # Who/what made the assessment
    context: Dict[str, Any]

@dataclass
class SkillGap:
    """Identified skill gap"""
    skill: SalesSkill
    current_score: float
    target_score: float
    gap_size: float
    severity: GapSeverity
    impact_on_performance: float  # How much this gap affects overall performance
    priority_rank: int  # 1 = highest priority
    root_causes: List[str]
    improvement_potential: float  # Expected improvement if gap is closed
    time_to_improve_weeks: int
    related_skills: List[SalesSkill]  # Skills that might improve together

@dataclass
class SkillProfile:
    """Complete skill profile for a sales rep"""
    rep_id: str
    assessment_date: datetime
    skill_assessments: List[SkillAssessment]
    identified_gaps: List[SkillGap]
    overall_skill_score: float
    top_strengths: List[SalesSkill]
    critical_weaknesses: List[SalesSkill]
    improvement_priorities: List[SkillGap]
    development_timeline: Dict[str, Any]

@dataclass
class CompetencyFramework:
    """Competency framework for different roles and experience levels"""
    role: str
    experience_level: str
    required_skills: Dict[SalesSkill, float]  # Skill -> minimum required score
    preferred_skills: Dict[SalesSkill, float]  # Skill -> preferred score
    critical_skills: List[SalesSkill]  # Skills that are critical for success
    skill_weights: Dict[SalesSkill, float]  # Relative importance of each skill

class SkillGapAnalysisEngine:
    """Advanced skill gap analysis and improvement planning system"""
    
    def __init__(self, config_path: str = "config/skill_analysis_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Skill profiles storage
        self.skill_profiles: Dict[str, SkillProfile] = {}
        self.competency_frameworks: Dict[str, CompetencyFramework] = {}
        
        # Analysis patterns and rules
        self.skill_indicators = self._initialize_skill_indicators()
        self.gap_impact_matrix = self._initialize_gap_impact_matrix()
        
        # Initialize competency frameworks
        self._initialize_competency_frameworks()
        
        logger.info("SkillGapAnalysisEngine initialized")
    
    async def assess_sales_rep_skills(
        self,
        rep_id: str,
        assessment_data: Dict[str, Any],
        assessment_method: AnalysisMethod = AnalysisMethod.CALL_ANALYSIS
    ) -> SkillProfile:
        """Assess all skills for a sales representative"""
        try:
            skill_assessments = []
            
            # Assess each skill based on available data
            for skill in SalesSkill:
                assessment = await self._assess_individual_skill(
                    rep_id, skill, assessment_data, assessment_method
                )
                if assessment:
                    skill_assessments.append(assessment)
            
            # Calculate overall skill score
            overall_score = self._calculate_overall_skill_score(skill_assessments)
            
            # Identify skill gaps
            gaps = await self._identify_skill_gaps(rep_id, skill_assessments)
            
            # Determine strengths and weaknesses
            strengths, weaknesses = self._categorize_strengths_weaknesses(skill_assessments)
            
            # Prioritize gaps
            prioritized_gaps = self._prioritize_gaps(gaps)
            
            # Create development timeline
            timeline = self._create_development_timeline(prioritized_gaps)
            
            # Create skill profile
            skill_profile = SkillProfile(
                rep_id=rep_id,
                assessment_date=datetime.now(),
                skill_assessments=skill_assessments,
                identified_gaps=gaps,
                overall_skill_score=overall_score,
                top_strengths=strengths[:5],  # Top 5 strengths
                critical_weaknesses=weaknesses[:3],  # Top 3 weaknesses
                improvement_priorities=prioritized_gaps[:5],  # Top 5 priorities
                development_timeline=timeline
            )
            
            # Store the profile
            self.skill_profiles[rep_id] = skill_profile
            
            logger.info(f"Completed skill assessment for rep {rep_id}")
            return skill_profile
            
        except Exception as e:
            logger.error(f"Error assessing skills for rep {rep_id}: {e}")
            raise
    
    async def analyze_call_for_skills(
        self,
        rep_id: str,
        call_transcript: str,
        call_metadata: Dict[str, Any],
        call_outcome: str
    ) -> Dict[SalesSkill, float]:
        """Analyze a sales call to extract skill performance indicators"""
        try:
            skill_scores = {}
            
            # Analyze transcript for each skill
            for skill in SalesSkill:
                score = await self._analyze_transcript_for_skill(
                    call_transcript, call_metadata, call_outcome, skill
                )
                if score is not None:
                    skill_scores[skill] = score
            
            logger.debug(f"Analyzed call for {len(skill_scores)} skills")
            return skill_scores
            
        except Exception as e:
            logger.error(f"Error analyzing call for skills: {e}")
            return {}
    
    async def get_skill_gap_report(
        self,
        rep_id: str,
        include_action_plan: bool = True,
        include_peer_comparison: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive skill gap report"""
        try:
            if rep_id not in self.skill_profiles:
                return {"error": "No skill profile found for rep"}
            
            profile = self.skill_profiles[rep_id]
            
            report = {
                "rep_id": rep_id,
                "assessment_date": profile.assessment_date.isoformat(),
                "overall_skill_score": profile.overall_skill_score,
                "skill_level_summary": self._create_skill_level_summary(profile),
                "critical_gaps": [
                    self._gap_to_dict(gap) for gap in profile.improvement_priorities
                ],
                "top_strengths": [skill.value for skill in profile.top_strengths],
                "critical_weaknesses": [skill.value for skill in profile.critical_weaknesses],
                "development_timeline": profile.development_timeline
            }
            
            # Add detailed skill breakdown
            report["skill_breakdown"] = {}
            for assessment in profile.skill_assessments:
                report["skill_breakdown"][assessment.skill.value] = {
                    "current_score": assessment.current_level,
                    "skill_level": assessment.skill_level_category.value,
                    "confidence": assessment.confidence,
                    "assessment_method": assessment.assessment_method.value,
                    "evidence": assessment.evidence[:3]  # Top 3 pieces of evidence
                }
            
            # Add action plan if requested
            if include_action_plan:
                report["action_plan"] = await self._create_skill_development_action_plan(rep_id)
            
            # Add peer comparison if requested
            if include_peer_comparison:
                report["peer_comparison"] = await self._create_peer_skill_comparison(rep_id)
            
            logger.info(f"Generated skill gap report for rep {rep_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating skill gap report: {e}")
            return {"error": str(e)}
    
    async def compare_team_skills(
        self,
        team_reps: List[str],
        focus_skills: List[SalesSkill] = None
    ) -> Dict[str, Any]:
        """Compare skills across a team of sales reps"""
        try:
            if focus_skills is None:
                focus_skills = list(SalesSkill)
            
            team_analysis = {
                "team_size": len(team_reps),
                "analysis_date": datetime.now().isoformat(),
                "skill_averages": {},
                "skill_distributions": {},
                "team_strengths": [],
                "team_weaknesses": [],
                "skill_correlation_matrix": {},
                "improvement_opportunities": []
            }
            
            # Calculate team averages and distributions for each skill
            for skill in focus_skills:
                scores = []
                for rep_id in team_reps:
                    if rep_id in self.skill_profiles:
                        score = self._get_skill_score(rep_id, skill)
                        if score is not None:
                            scores.append(score)
                
                if scores:
                    team_analysis["skill_averages"][skill.value] = {
                        "mean": statistics.mean(scores),
                        "median": statistics.median(scores),
                        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                        "min": min(scores),
                        "max": max(scores),
                        "range": max(scores) - min(scores)
                    }
                    
                    team_analysis["skill_distributions"][skill.value] = {
                        "expert": len([s for s in scores if s >= 91]),
                        "proficient": len([s for s in scores if 76 <= s < 91]),
                        "competent": len([s for s in scores if 51 <= s < 76]),
                        "developing": len([s for s in scores if 26 <= s < 51]),
                        "novice": len([s for s in scores if s < 26])
                    }
            
            # Identify team strengths (high average scores)
            skill_averages = [
                (skill, data["mean"]) for skill, data in team_analysis["skill_averages"].items()
            ]
            team_analysis["team_strengths"] = [
                skill for skill, avg in sorted(skill_averages, key=lambda x: x[1], reverse=True)[:5]
            ]
            
            # Identify team weaknesses (low average scores or high variation)
            team_analysis["team_weaknesses"] = [
                skill for skill, avg in sorted(skill_averages, key=lambda x: x[1])[:5]
            ]
            
            # Calculate improvement opportunities
            team_analysis["improvement_opportunities"] = await self._identify_team_improvement_opportunities(
                team_reps, focus_skills
            )
            
            logger.info(f"Completed team skill analysis for {len(team_reps)} reps")
            return team_analysis
            
        except Exception as e:
            logger.error(f"Error comparing team skills: {e}")
            return {"error": str(e)}
    
    async def track_skill_improvement(
        self,
        rep_id: str,
        skill: SalesSkill,
        timeframe_weeks: int = 12
    ) -> Dict[str, Any]:
        """Track skill improvement over time"""
        try:
            # This would typically query historical assessment data
            # For now, we'll simulate progress tracking
            
            improvement_data = {
                "rep_id": rep_id,
                "skill": skill.value,
                "timeframe_weeks": timeframe_weeks,
                "baseline_score": 0,
                "current_score": 0,
                "target_score": 0,
                "progress_percentage": 0,
                "improvement_rate": 0,  # Points per week
                "projected_completion": None,
                "milestones_achieved": [],
                "recommended_actions": []
            }
            
            # Get current skill assessment
            if rep_id in self.skill_profiles:
                current_assessment = self._get_skill_assessment(rep_id, skill)
                if current_assessment:
                    improvement_data["current_score"] = current_assessment.current_level
                    
                    # Find relevant gap for target
                    for gap in self.skill_profiles[rep_id].identified_gaps:
                        if gap.skill == skill:
                            improvement_data["target_score"] = gap.target_score
                            improvement_data["baseline_score"] = gap.current_score
                            
                            # Calculate progress
                            total_improvement_needed = gap.target_score - gap.current_score
                            if total_improvement_needed > 0:
                                current_improvement = improvement_data["current_score"] - improvement_data["baseline_score"]
                                improvement_data["progress_percentage"] = (current_improvement / total_improvement_needed) * 100
                            
                            break
            
            return improvement_data
            
        except Exception as e:
            logger.error(f"Error tracking skill improvement: {e}")
            return {"error": str(e)}
    
    async def _assess_individual_skill(
        self,
        rep_id: str,
        skill: SalesSkill,
        assessment_data: Dict[str, Any],
        method: AnalysisMethod
    ) -> Optional[SkillAssessment]:
        """Assess a single skill for a sales rep"""
        try:
            indicators = self.skill_indicators.get(skill, {})
            score = 0.0
            evidence = []
            confidence = 0.5
            
            if method == AnalysisMethod.CALL_ANALYSIS:
                score, evidence, confidence = await self._assess_skill_from_calls(
                    skill, assessment_data, indicators
                )
            elif method == AnalysisMethod.PERFORMANCE_METRICS:
                score, evidence, confidence = await self._assess_skill_from_metrics(
                    skill, assessment_data, indicators
                )
            elif method == AnalysisMethod.MANAGER_ASSESSMENT:
                score, evidence, confidence = await self._assess_skill_from_manager_feedback(
                    skill, assessment_data, indicators
                )
            
            if score > 0:
                skill_level = self._determine_skill_level(score)
                
                return SkillAssessment(
                    skill=skill,
                    current_level=score,
                    skill_level_category=skill_level,
                    assessment_method=method,
                    confidence=confidence,
                    evidence=evidence,
                    assessment_date=datetime.now(),
                    assessor=f"system_{method.value}",
                    context={"method_data": assessment_data.get(skill.value, {})}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error assessing skill {skill.value}: {e}")
            return None
    
    async def _assess_skill_from_calls(
        self,
        skill: SalesSkill,
        call_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Tuple[float, List[str], float]:
        """Assess skill based on call analysis data"""
        try:
            score = 0.0
            evidence = []
            confidence = 0.5
            
            calls = call_data.get("calls", [])
            if not calls:
                return 0.0, [], 0.0
            
            skill_scores = []
            
            for call in calls:
                call_score = 0.0
                call_evidence = []
                
                transcript = call.get("transcript", "")
                metadata = call.get("metadata", {})
                outcome = call.get("outcome", "unknown")
                
                # Skill-specific analysis
                if skill == SalesSkill.QUESTIONING_TECHNIQUE:
                    questions = self._count_questions(transcript)
                    open_questions = self._count_open_questions(transcript)
                    if questions > 0:
                        call_score = min(100, (open_questions / questions) * 100)
                        call_evidence.append(f"Asked {open_questions} open questions out of {questions} total")
                
                elif skill == SalesSkill.OBJECTION_HANDLING:
                    objections = self._detect_objections(transcript)
                    responses = self._detect_objection_responses(transcript)
                    if objections > 0:
                        response_rate = responses / objections
                        call_score = min(100, response_rate * 80)  # Max 80 points for perfect response rate
                        call_evidence.append(f"Responded to {responses}/{objections} objections")
                
                elif skill == SalesSkill.CLOSING_TECHNIQUES:
                    closing_attempts = self._detect_closing_attempts(transcript)
                    successful_close = outcome in ["won", "closed", "sold"]
                    call_score = min(100, closing_attempts * 20 + (50 if successful_close else 0))
                    call_evidence.append(f"Made {closing_attempts} closing attempts")
                
                elif skill == SalesSkill.RAPPORT_BUILDING:
                    rapport_indicators = self._detect_rapport_building(transcript)
                    call_score = min(100, len(rapport_indicators) * 15)
                    call_evidence.extend(rapport_indicators[:3])
                
                elif skill == SalesSkill.PRODUCT_KNOWLEDGE:
                    product_mentions = self._detect_product_knowledge(transcript)
                    technical_accuracy = self._assess_technical_accuracy(transcript)
                    call_score = min(100, (product_mentions * 10) + (technical_accuracy * 50))
                    call_evidence.append(f"Demonstrated product knowledge {product_mentions} times")
                
                # Add more skill assessments...
                
                if call_score > 0:
                    skill_scores.append(call_score)
                    evidence.extend(call_evidence[:2])  # Take top 2 pieces of evidence per call
            
            if skill_scores:
                score = statistics.mean(skill_scores)
                confidence = min(0.9, 0.5 + (len(skill_scores) * 0.1))  # Higher confidence with more data
            
            return score, evidence[:5], confidence  # Return top 5 pieces of evidence
            
        except Exception as e:
            logger.error(f"Error assessing skill from calls: {e}")
            return 0.0, [], 0.0
    
    async def _assess_skill_from_metrics(
        self,
        skill: SalesSkill,
        metrics_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Tuple[float, List[str], float]:
        """Assess skill based on performance metrics"""
        try:
            score = 0.0
            evidence = []
            confidence = 0.7  # Metrics are generally reliable
            
            # Map skills to performance metrics
            if skill == SalesSkill.CLOSING_TECHNIQUES:
                conversion_rate = metrics_data.get("conversion_rate", 0)
                score = min(100, conversion_rate * 100)
                evidence.append(f"Conversion rate: {conversion_rate:.1%}")
                
            elif skill == SalesSkill.FOLLOW_UP:
                follow_up_rate = metrics_data.get("follow_up_consistency", 0)
                response_time = metrics_data.get("average_response_time_hours", 24)
                score = min(100, follow_up_rate * 80 + max(0, (48 - response_time) / 48) * 20)
                evidence.append(f"Follow-up consistency: {follow_up_rate:.1%}")
                
            elif skill == SalesSkill.PIPELINE_MANAGEMENT:
                pipeline_accuracy = metrics_data.get("pipeline_accuracy", 0)
                forecast_accuracy = metrics_data.get("forecast_accuracy", 0)
                score = min(100, (pipeline_accuracy + forecast_accuracy) / 2 * 100)
                evidence.append(f"Pipeline accuracy: {pipeline_accuracy:.1%}")
                
            elif skill == SalesSkill.TIME_MANAGEMENT:
                activity_volume = metrics_data.get("daily_activities", 0)
                target_activities = 20  # Assume target
                score = min(100, (activity_volume / target_activities) * 100)
                evidence.append(f"Daily activities: {activity_volume} (target: {target_activities})")
            
            return score, evidence, confidence
            
        except Exception as e:
            logger.error(f"Error assessing skill from metrics: {e}")
            return 0.0, [], 0.0
    
    async def _assess_skill_from_manager_feedback(
        self,
        skill: SalesSkill,
        feedback_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Tuple[float, List[str], float]:
        """Assess skill based on manager assessment"""
        try:
            manager_scores = feedback_data.get("manager_assessments", {})
            skill_score = manager_scores.get(skill.value, 0)
            
            evidence = []
            if "manager_comments" in feedback_data:
                comments = feedback_data["manager_comments"].get(skill.value, [])
                evidence.extend(comments[:3])
            
            confidence = 0.8  # Manager assessments are usually reliable
            
            return float(skill_score), evidence, confidence
            
        except Exception as e:
            logger.error(f"Error assessing skill from manager feedback: {e}")
            return 0.0, [], 0.0
    
    async def _identify_skill_gaps(
        self,
        rep_id: str,
        skill_assessments: List[SkillAssessment]
    ) -> List[SkillGap]:
        """Identify skill gaps by comparing against competency framework"""
        try:
            gaps = []
            
            # Get rep's competency framework (based on role/experience)
            framework = self._get_competency_framework(rep_id)
            if not framework:
                logger.warning(f"No competency framework found for rep {rep_id}")
                return gaps
            
            for assessment in skill_assessments:
                skill = assessment.skill
                current_score = assessment.current_level
                
                # Determine target score based on framework
                target_score = framework.required_skills.get(skill, 60.0)  # Default to 60
                if skill in framework.preferred_skills:
                    target_score = framework.preferred_skills[skill]
                
                # Calculate gap
                gap_size = target_score - current_score
                
                if gap_size > 5:  # Only consider significant gaps
                    severity = self._determine_gap_severity(gap_size)
                    impact = self._calculate_impact_on_performance(skill, gap_size, framework)
                    priority = self._calculate_gap_priority(skill, gap_size, impact, framework)
                    
                    gap = SkillGap(
                        skill=skill,
                        current_score=current_score,
                        target_score=target_score,
                        gap_size=gap_size,
                        severity=severity,
                        impact_on_performance=impact,
                        priority_rank=priority,
                        root_causes=self._identify_root_causes(skill, assessment),
                        improvement_potential=self._estimate_improvement_potential(skill, gap_size),
                        time_to_improve_weeks=self._estimate_improvement_time(skill, gap_size),
                        related_skills=self._find_related_skills(skill)
                    )
                    
                    gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying skill gaps: {e}")
            return []
    
    def _determine_skill_level(self, score: float) -> SkillLevel:
        """Determine skill level category based on score"""
        if score >= 91:
            return SkillLevel.EXPERT
        elif score >= 76:
            return SkillLevel.PROFICIENT
        elif score >= 51:
            return SkillLevel.COMPETENT
        elif score >= 26:
            return SkillLevel.DEVELOPING
        else:
            return SkillLevel.NOVICE
    
    def _determine_gap_severity(self, gap_size: float) -> GapSeverity:
        """Determine gap severity based on gap size"""
        if gap_size >= 40:
            return GapSeverity.CRITICAL
        elif gap_size >= 25:
            return GapSeverity.HIGH
        elif gap_size >= 15:
            return GapSeverity.MEDIUM
        elif gap_size >= 5:
            return GapSeverity.LOW
        else:
            return GapSeverity.MINIMAL
    
    # Utility methods for text analysis
    def _count_questions(self, transcript: str) -> int:
        """Count total questions in transcript"""
        return transcript.count('?')
    
    def _count_open_questions(self, transcript: str) -> int:
        """Count open-ended questions"""
        open_question_patterns = [
            r'\bwhat\b.*\?', r'\bhow\b.*\?', r'\bwhy\b.*\?',
            r'\bwhere\b.*\?', r'\bwhen\b.*\?', r'\bwho\b.*\?',
            r'\btell me\b.*\?', r'\bdescribe\b.*\?', r'\bexplain\b.*\?'
        ]
        
        count = 0
        for pattern in open_question_patterns:
            count += len(re.findall(pattern, transcript, re.IGNORECASE))
        
        return count
    
    def _detect_objections(self, transcript: str) -> int:
        """Detect objections in customer speech"""
        objection_patterns = [
            r'too expensive', r'budget', r'think about it', r'not interested',
            r'need to discuss', r'already have', r'not sure', r'concerned'
        ]
        
        count = 0
        for pattern in objection_patterns:
            count += len(re.findall(pattern, transcript, re.IGNORECASE))
        
        return count
    
    def _detect_objection_responses(self, transcript: str) -> int:
        """Detect responses to objections"""
        response_patterns = [
            r'understand your concern', r'let me explain', r'i see',
            r'that\'s a common', r'appreciate you sharing', r'feel.*felt.*found'
        ]
        
        count = 0
        for pattern in response_patterns:
            count += len(re.findall(pattern, transcript, re.IGNORECASE))
        
        return count
    
    def _detect_closing_attempts(self, transcript: str) -> int:
        """Detect closing attempts"""
        closing_patterns = [
            r'ready to move forward', r'shall we proceed', r'would you like to',
            r'when can we start', r'are you interested in', r'next steps'
        ]
        
        count = 0
        for pattern in closing_patterns:
            count += len(re.findall(pattern, transcript, re.IGNORECASE))
        
        return count
    
    def _detect_rapport_building(self, transcript: str) -> List[str]:
        """Detect rapport building indicators"""
        rapport_indicators = []
        
        patterns = {
            'personal_connection': r'(family|weekend|hobby|vacation)',
            'empathy': r'(understand|appreciate|imagine)',
            'agreement': r'(absolutely|exactly|definitely)',
            'humor': r'(haha|funny|laugh)'
        }
        
        for indicator_type, pattern in patterns.items():
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            if matches:
                rapport_indicators.append(f"{indicator_type}: {len(matches)} instances")
        
        return rapport_indicators
    
    def _detect_product_knowledge(self, transcript: str) -> int:
        """Detect product knowledge demonstrations"""
        product_terms = [
            'feature', 'functionality', 'capability', 'integration',
            'API', 'dashboard', 'analytics', 'reporting', 'automation'
        ]
        
        count = 0
        for term in product_terms:
            count += len(re.findall(rf'\b{term}\b', transcript, re.IGNORECASE))
        
        return count
    
    def _assess_technical_accuracy(self, transcript: str) -> float:
        """Assess technical accuracy (simplified)"""
        # This would involve more sophisticated analysis in practice
        technical_terms = transcript.count('technical') + transcript.count('specification')
        incorrect_terms = transcript.count('um') + transcript.count('not sure')
        
        if technical_terms + incorrect_terms == 0:
            return 0.5  # Neutral score
        
        return technical_terms / (technical_terms + incorrect_terms)
    
    # Additional helper methods would be implemented here...
    
    def _load_config(self) -> Dict[str, Any]:
        """Load skill analysis configuration"""
        return {
            "assessment_confidence_threshold": 0.6,
            "minimum_evidence_count": 2,
            "gap_significance_threshold": 5.0,
            "improvement_timeline_weeks": 12
        }
    
    def _initialize_skill_indicators(self) -> Dict[SalesSkill, Dict[str, Any]]:
        """Initialize indicators for each skill"""
        # This would be a comprehensive mapping in practice
        return {
            SalesSkill.QUESTIONING_TECHNIQUE: {
                "patterns": ["what", "how", "why", "tell me", "describe"],
                "weight": 0.8
            },
            SalesSkill.OBJECTION_HANDLING: {
                "patterns": ["understand", "appreciate", "let me explain"],
                "weight": 0.9
            }
            # More skills would be mapped here...
        }
    
    def _initialize_gap_impact_matrix(self) -> Dict[SalesSkill, float]:
        """Initialize impact weights for different skills"""
        return {
            SalesSkill.CLOSING_TECHNIQUES: 0.9,
            SalesSkill.OBJECTION_HANDLING: 0.8,
            SalesSkill.PRODUCT_KNOWLEDGE: 0.7,
            SalesSkill.RAPPORT_BUILDING: 0.6,
            # More skills...
        }
    
    def _initialize_competency_frameworks(self):
        """Initialize competency frameworks for different roles"""
        # Example framework for Senior Sales Rep
        senior_framework = CompetencyFramework(
            role="senior_sales_rep",
            experience_level="senior",
            required_skills={
                SalesSkill.CLOSING_TECHNIQUES: 80,
                SalesSkill.OBJECTION_HANDLING: 75,
                SalesSkill.PRODUCT_KNOWLEDGE: 85,
                SalesSkill.RAPPORT_BUILDING: 70,
                SalesSkill.QUESTIONING_TECHNIQUE: 75
            },
            preferred_skills={
                SalesSkill.CLOSING_TECHNIQUES: 90,
                SalesSkill.PRODUCT_KNOWLEDGE: 95
            },
            critical_skills=[
                SalesSkill.CLOSING_TECHNIQUES,
                SalesSkill.OBJECTION_HANDLING,
                SalesSkill.PRODUCT_KNOWLEDGE
            ],
            skill_weights={
                SalesSkill.CLOSING_TECHNIQUES: 0.3,
                SalesSkill.OBJECTION_HANDLING: 0.25,
                SalesSkill.PRODUCT_KNOWLEDGE: 0.2,
                SalesSkill.RAPPORT_BUILDING: 0.15,
                SalesSkill.QUESTIONING_TECHNIQUE: 0.1
            }
        )
        
        self.competency_frameworks["senior_sales_rep"] = senior_framework
        
        # More frameworks would be added here...


# Global instance
skill_gap_analysis_engine = SkillGapAnalysisEngine()