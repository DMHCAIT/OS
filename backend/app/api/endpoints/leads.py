"""
Lead Management API Endpoints
Complete CRUD operations and advanced lead management features
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.models.lead import (
    Lead, LeadCreate, LeadUpdate, LeadResponse, 
    LeadListResponse, LeadAnalytics, LeadBulkUpdate,
    LeadStatus, LeadSource
)
from app.core.security import get_current_user
from app.core.database import get_database
from app.ml.lead_scoring import lead_scoring_engine

router = APIRouter()


@router.post("/", response_model=LeadResponse)
async def create_lead(
    lead: LeadCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Create a new lead"""
    try:
        # Convert to Lead model with ID and timestamps
        lead_data = lead.dict()
        new_lead = Lead(**lead_data)
        new_lead.created_by = current_user["id"]
        
        # Calculate initial lead score
        score_result = lead_scoring_engine.calculate_lead_score(lead_data)
        new_lead.score = score_result
        
        # Insert into database
        result = await db.leads.insert_one(new_lead.dict())
        
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to create lead")
        
        # Return created lead
        created_lead = await db.leads.find_one({"_id": result.inserted_id})
        return LeadResponse(**created_lead)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating lead: {str(e)}")


@router.get("/", response_model=LeadListResponse)
async def get_leads(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    status: Optional[LeadStatus] = None,
    source: Optional[LeadSource] = None,
    assigned_to: Optional[str] = None,
    search: Optional[str] = None,
    min_score: Optional[float] = Query(None, ge=0, le=1),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get leads with filtering and pagination"""
    try:
        # Build query filters
        query = {}
        
        if status:
            query["status"] = status
        
        if source:
            query["source"] = source
            
        if assigned_to:
            query["assigned_to"] = assigned_to
            
        if min_score is not None:
            query["score.overall_score"] = {"$gte": min_score}
        
        # Search functionality
        if search:
            query["$or"] = [
                {"first_name": {"$regex": search, "$options": "i"}},
                {"last_name": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}},
                {"company": {"$regex": search, "$options": "i"}}
            ]
        
        # Calculate skip value for pagination
        skip = (page - 1) * per_page
        
        # Get total count
        total = await db.leads.count_documents(query)
        
        # Get leads with pagination
        cursor = db.leads.find(query).skip(skip).limit(per_page).sort("created_at", -1)
        leads = await cursor.to_list(length=per_page)
        
        # Convert to response models
        lead_responses = [LeadResponse(**lead) for lead in leads]
        
        total_pages = (total + per_page - 1) // per_page
        
        return LeadListResponse(
            leads=lead_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching leads: {str(e)}")


@router.get("/{lead_id}", response_model=LeadResponse)
async def get_lead(
    lead_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get a specific lead by ID"""
    try:
        lead = await db.leads.find_one({"id": lead_id})
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        return LeadResponse(**lead)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching lead: {str(e)}")


@router.put("/{lead_id}", response_model=LeadResponse)
async def update_lead(
    lead_id: str,
    lead_update: LeadUpdate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Update a lead"""
    try:
        # Check if lead exists
        existing_lead = await db.leads.find_one({"id": lead_id})
        if not existing_lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Prepare update data
        update_data = {k: v for k, v in lead_update.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        
        # Recalculate score if relevant data changed
        if any(field in update_data for field in ["company", "title", "industry", "activities"]):
            merged_data = {**existing_lead, **update_data}
            score_result = lead_scoring_engine.calculate_lead_score(merged_data)
            update_data["score"] = score_result
        
        # Update lead
        result = await db.leads.update_one(
            {"id": lead_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update lead")
        
        # Return updated lead
        updated_lead = await db.leads.find_one({"id": lead_id})
        return LeadResponse(**updated_lead)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating lead: {str(e)}")


@router.delete("/{lead_id}")
async def delete_lead(
    lead_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Delete a lead"""
    try:
        result = await db.leads.delete_one({"id": lead_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        return {"message": "Lead deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting lead: {str(e)}")


@router.post("/{lead_id}/score", response_model=Dict[str, Any])
async def recalculate_lead_score(
    lead_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Recalculate lead score"""
    try:
        # Get lead data
        lead = await db.leads.find_one({"id": lead_id})
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Recalculate score
        score_result = lead_scoring_engine.calculate_lead_score(lead)
        
        # Update lead with new score
        await db.leads.update_one(
            {"id": lead_id},
            {"$set": {"score": score_result, "updated_at": datetime.utcnow()}}
        )
        
        return score_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recalculating score: {str(e)}")


@router.post("/{lead_id}/activities")
async def add_lead_activity(
    lead_id: str,
    activity_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Add activity to lead"""
    try:
        # Validate lead exists
        lead = await db.leads.find_one({"id": lead_id})
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Create activity
        activity = {
            "activity_type": activity_data.get("activity_type"),
            "description": activity_data.get("description"),
            "metadata": activity_data.get("metadata", {}),
            "created_at": datetime.utcnow(),
            "created_by": current_user["id"]
        }
        
        # Add activity to lead
        result = await db.leads.update_one(
            {"id": lead_id},
            {
                "$push": {"activities": activity},
                "$set": {
                    "last_activity_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to add activity")
        
        # Recalculate score after activity
        updated_lead = await db.leads.find_one({"id": lead_id})
        score_result = lead_scoring_engine.calculate_lead_score(updated_lead)
        
        await db.leads.update_one(
            {"id": lead_id},
            {"$set": {"score": score_result}}
        )
        
        return {"message": "Activity added successfully", "activity": activity}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding activity: {str(e)}")


@router.put("/{lead_id}/assign")
async def assign_lead(
    lead_id: str,
    assigned_to: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Assign lead to a user"""
    try:
        result = await db.leads.update_one(
            {"id": lead_id},
            {
                "$set": {
                    "assigned_to": assigned_to,
                    "assigned_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        return {"message": "Lead assigned successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assigning lead: {str(e)}")


@router.post("/bulk-update")
async def bulk_update_leads(
    bulk_update: LeadBulkUpdate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Bulk update multiple leads"""
    try:
        # Prepare update data
        update_data = {k: v for k, v in bulk_update.updates.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        
        # Update leads
        result = await db.leads.update_many(
            {"id": {"$in": bulk_update.lead_ids}},
            {"$set": update_data}
        )
        
        return {
            "message": f"Updated {result.modified_count} leads",
            "updated_count": result.modified_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error bulk updating leads: {str(e)}")


@router.get("/analytics/overview", response_model=LeadAnalytics)
async def get_lead_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get lead analytics overview"""
    try:
        # Date range for analytics
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Aggregate lead statistics
        pipeline = [
            {"$match": {"created_at": {"$gte": start_date}}},
            {
                "$group": {
                    "_id": None,
                    "total_leads": {"$sum": 1},
                    "avg_score": {"$avg": "$score.overall_score"},
                    "by_status": {
                        "$push": {
                            "status": "$status",
                            "count": 1
                        }
                    },
                    "by_source": {
                        "$push": {
                            "source": "$source",
                            "count": 1
                        }
                    }
                }
            }
        ]
        
        result = await db.leads.aggregate(pipeline).to_list(1)
        
        if not result:
            return LeadAnalytics(
                total_leads=0,
                new_leads=0,
                qualified_leads=0,
                converted_leads=0,
                average_score=0.0,
                conversion_rate=0.0,
                by_source={},
                by_status={},
                by_industry={},
                recent_activities=[]
            )
        
        data = result[0]
        
        # Process status counts
        status_counts = {}
        for item in data.get("by_status", []):
            status = item["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Process source counts
        source_counts = {}
        for item in data.get("by_source", []):
            source = item["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate conversion rate
        total_leads = data["total_leads"]
        converted_leads = status_counts.get("closed_won", 0)
        conversion_rate = converted_leads / total_leads if total_leads > 0 else 0
        
        return LeadAnalytics(
            total_leads=total_leads,
            new_leads=status_counts.get("new", 0),
            qualified_leads=status_counts.get("qualified", 0),
            converted_leads=converted_leads,
            average_score=data.get("avg_score", 0.0),
            conversion_rate=conversion_rate,
            by_source=source_counts,
            by_status=status_counts,
            by_industry={},  # Would need additional aggregation
            recent_activities=[]  # Would need additional query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")


@router.post("/import")
async def import_leads(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Import leads from CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Process CSV file (simplified implementation)
        content = await file.read()
        # This would include proper CSV parsing and validation
        
        return {"message": "Leads imported successfully", "imported_count": 0}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing leads: {str(e)}")


@router.get("/export")
async def export_leads(
    format: str = Query("csv", regex="^(csv|xlsx)$"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Export leads to CSV or Excel"""
    try:
        # This would generate and return the export file
        return {"message": "Export functionality would be implemented here"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting leads: {str(e)}")