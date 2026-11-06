#!/usr/bin/env python3
"""
SQLModel with Instructor - Comprehensive Example

This example demonstrates AI-powered database operations with advanced patterns.

Requirements:
    pip install instructor sqlmodel openai

Usage:
    python run.py

Note: Make sure to set your OPENAI_API_KEY environment variable.
"""

import asyncio
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Optional
from uuid import UUID, uuid4

import instructor
from openai import AsyncOpenAI, OpenAI
from pydantic import validator
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
sync_client = instructor.from_openai(OpenAI())
async_client = instructor.from_openai(AsyncOpenAI())

# Database setup
engine = create_engine("sqlite:///heroes_demo.db", echo=False)


# Performance monitoring decorator
def monitor_ai_calls(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"AI call took {duration:.2f} seconds")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"AI call took {duration:.2f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Models with relationships and advanced patterns
class Team(SQLModel, table=True):
    """Team model with relationship to heroes"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=2, max_length=50)
    city: str = Field(min_length=2, max_length=50)
    founded_year: Optional[int] = Field(default=None, ge=1900, le=2024)

    # Relationship to heroes
    heroes: list["Hero"] = Relationship(back_populates="team")


class Hero(SQLModel, instructor.ResponseSchema, table=True):
    """Hero model with auto-generated fields and validation"""

    __table_args__ = {"extend_existing": True}

    # Auto-generated fields excluded from AI generation
    id: SkipJsonSchema[Optional[int]] = Field(default=None, primary_key=True)
    created_at: SkipJsonSchema[datetime] = Field(default_factory=datetime.utcnow)
    uuid: SkipJsonSchema[UUID] = Field(default_factory=uuid4)

    # AI-generated fields with validation
    name: str = Field(min_length=2, max_length=50, description="Hero's public name")
    secret_name: str = Field(
        min_length=2, max_length=50, description="Hero's secret identity"
    )
    age: Optional[int] = Field(default=None, ge=16, le=100, description="Hero's age")
    power_level: int = Field(ge=1, le=100, description="Power level from 1-100")
    origin_story: str = Field(
        min_length=10, max_length=200, description="Brief origin story"
    )

    # Foreign key relationship
    team_id: SkipJsonSchema[Optional[int]] = Field(default=None, foreign_key="team.id")
    team: Optional[Team] = Relationship(back_populates="heroes")

    @validator("name")
    def validate_name_format(cls, v):
        """Ensure hero name doesn't contain inappropriate words"""
        forbidden_words = ["villain", "evil", "bad"]
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError(f"Hero name cannot contain: {', '.join(forbidden_words)}")
        return v


class Product(SQLModel, instructor.ResponseSchema, table=True):
    """Product model demonstrating different AI generation patterns"""

    __table_args__ = {"extend_existing": True}

    # Auto-generated fields
    id: SkipJsonSchema[UUID] = Field(default_factory=uuid4, primary_key=True)
    created_at: SkipJsonSchema[datetime] = Field(default_factory=datetime.utcnow)

    # AI-generated fields
    name: str = Field(description="Product name")
    description: str = Field(description="Detailed product description")
    price: float = Field(gt=0, description="Product price in USD")
    category: str = Field(description="Product category")


# Functions for AI data generation
@monitor_ai_calls
def create_hero(prompt: str = "Create a unique superhero") -> Hero:
    """Generate a single hero using AI"""
    try:
        return sync_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Hero,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error(f"Failed to create hero: {str(e)}")
        raise


@monitor_ai_calls
async def create_hero_async(prompt: str = "Create a unique superhero") -> Hero:
    """Generate a single hero using AI (async)"""
    try:
        return await async_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Hero,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error(f"Failed to create hero: {str(e)}")
        raise


@monitor_ai_calls
async def create_hero_team_async(team_size: int = 5) -> list[Hero]:
    """Generate multiple heroes concurrently"""
    try:
        return await async_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=list[Hero],
            messages=[
                {
                    "role": "user",
                    "content": f"Create a team of {team_size} diverse superheroes with different powers",
                },
            ],
            max_retries=3,
        )
    except Exception as e:
        logger.error(f"Failed to create hero team: {str(e)}")
        raise


async def create_heroes_batch(prompts: list[str]) -> list[Hero]:
    """Generate multiple heroes concurrently from different prompts"""
    tasks = []
    for prompt in prompts:
        task = create_hero_async(prompt)
        tasks.append(task)

    return await asyncio.gather(*tasks, return_exceptions=True)


def create_product(category: str) -> Product:
    """Generate a product for a specific category"""
    try:
        return sync_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Product,
            messages=[
                {
                    "role": "user",
                    "content": f"Create a {category} product with realistic pricing",
                },
            ],
        )
    except Exception as e:
        logger.error(f"Failed to create product: {str(e)}")
        raise


# Database operations
def setup_database():
    """Create all tables"""
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def create_sample_teams():
    """Create sample teams for heroes to join"""
    teams_data = [
        {"name": "Justice League", "city": "Metropolis", "founded_year": 1960},
        {"name": "Avengers", "city": "New York", "founded_year": 1963},
        {"name": "X-Men", "city": "Westchester", "founded_year": 1963},
    ]

    with Session(engine) as session:
        for team_data in teams_data:
            # Check if team already exists
            existing_team = session.exec(
                select(Team).where(Team.name == team_data["name"])
            ).first()

            if not existing_team:
                team = Team(**team_data)
                session.add(team)

        session.commit()
        logger.info("Sample teams created")


def assign_hero_to_team(hero: Hero, team_name: str):
    """Assign a hero to a team"""
    with Session(engine) as session:
        # Get the team
        team = session.exec(select(Team).where(Team.name == team_name)).first()
        if team:
            hero.team_id = team.id
            session.add(hero)
            session.commit()
            session.refresh(hero)
            logger.info(f"Assigned {hero.name} to {team_name}")
        else:
            logger.warning(f"Team {team_name} not found")


def list_heroes_with_teams():
    """List all heroes with their team information"""
    with Session(engine) as session:
        statement = select(Hero, Team).join(Team, Hero.team_id == Team.id, isouter=True)
        results = session.exec(statement).all()

        logger.info("Heroes and their teams:")
        for hero, team in results:
            team_name = team.name if team else "No team"
            logger.info(
                f"- {hero.name} ({hero.secret_name}) - Power Level: {hero.power_level} - Team: {team_name}"
            )


def demonstrate_validation_errors():
    """Show how validation works with invalid data"""
    logger.info("Testing validation...")

    try:
        # This should fail due to validator
        Hero(
            name="Evil Villain",  # Contains forbidden word
            secret_name="Bad Guy",
            power_level=50,
            origin_story="A story of evil deeds",
        )
    except ValueError as e:
        logger.info(f"Validation caught invalid name: {e}")

    try:
        # This should fail due to field constraints
        Hero(
            name="Good Hero",
            secret_name="G",  # Too short
            power_level=150,  # Too high
            origin_story="Short",  # Too short
        )
    except ValueError as e:
        logger.info(f"Validation caught field constraint violation: {e}")


async def main():
    """Main demonstration function"""
    logger.info("Starting SQLModel with Instructor demonstration...")

    # Setup
    setup_database()
    create_sample_teams()

    # Demonstrate validation
    demonstrate_validation_errors()

    # 1. Basic hero creation
    logger.info("\n1. Creating a single hero...")
    hero1 = create_hero("Create a tech-based superhero")

    with Session(engine) as session:
        session.add(hero1)
        session.commit()
        session.refresh(hero1)

    logger.info(f"Created hero: {hero1.name} (Power Level: {hero1.power_level})")
    logger.info(f"Origin: {hero1.origin_story}")
    assign_hero_to_team(hero1, "Avengers")

    # 2. Async hero creation
    logger.info("\n2. Creating a hero asynchronously...")
    hero2 = await create_hero_async("Create a magic-based superhero")

    with Session(engine) as session:
        session.add(hero2)
        session.commit()
        session.refresh(hero2)

    logger.info(f"Created async hero: {hero2.name} (Power Level: {hero2.power_level})")
    assign_hero_to_team(hero2, "Justice League")

    # 3. Bulk hero creation
    logger.info("\n3. Creating a team of heroes...")
    hero_team = await create_hero_team_async(3)

    with Session(engine) as session:
        for hero in hero_team:
            session.add(hero)
        session.commit()

        for hero in hero_team:
            session.refresh(hero)

    logger.info(f"Created team of {len(hero_team)} heroes")
    for hero in hero_team:
        assign_hero_to_team(hero, "X-Men")

    # 4. Concurrent hero creation with different prompts
    logger.info("\n4. Creating heroes concurrently...")
    prompts = [
        "Create a fire-based superhero",
        "Create a water-based superhero",
        "Create an earth-based superhero",
        "Create a wind-based superhero",
    ]

    concurrent_heroes = await create_heroes_batch(prompts)

    with Session(engine) as session:
        for hero in concurrent_heroes:
            if isinstance(hero, Hero):  # Check if not an exception
                session.add(hero)
        session.commit()

    logger.info(
        f"Created {len([h for h in concurrent_heroes if isinstance(h, Hero)])} heroes concurrently"
    )

    # 5. Product creation (different model)
    logger.info("\n5. Creating products...")
    categories = ["electronics", "clothing", "books"]

    for category in categories:
        product = create_product(category)
        with Session(engine) as session:
            session.add(product)
            session.commit()
            session.refresh(product)

        logger.info(
            f"Created {category} product: {product.name} - ${product.price:.2f}"
        )

    # 6. Display results
    logger.info("\n6. Final results:")
    list_heroes_with_teams()

    # 7. Database statistics
    with Session(engine) as session:
        total_heroes = len(session.exec(select(Hero)).all())
        total_teams = len(session.exec(select(Team)).all())
        total_products = len(session.exec(select(Product)).all())

    logger.info(f"\nDatabase contains:")
    logger.info(f"- {total_heroes} heroes")
    logger.info(f"- {total_teams} teams")
    logger.info(f"- {total_products} products")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
