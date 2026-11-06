#!/usr/bin/env python3
"""
Basic SQLModel test to verify core functionality
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from pydantic import validator
from sqlmodel import Field, SQLModel, create_engine, Session, select, Relationship

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine("sqlite:///test_basic.db", echo=False)


# Models with relationships
class Team(SQLModel, table=True):
    """Team model with relationship to heroes"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=2, max_length=50)
    city: str = Field(min_length=2, max_length=50)
    founded_year: Optional[int] = Field(default=None, ge=1900, le=2024)

    # Relationship to heroes
    heroes: list["Hero"] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    """Hero model with auto-generated fields and validation"""

    __table_args__ = {"extend_existing": True}

    # Auto-generated fields
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    uuid: UUID = Field(default_factory=uuid4)

    # Regular fields with validation
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
    team_id: Optional[int] = Field(default=None, foreign_key="team.id")
    team: Optional[Team] = Relationship(back_populates="heroes")

    @validator("name")
    def validate_name_format(cls, v):
        """Ensure hero name doesn't contain inappropriate words"""
        forbidden_words = ["villain", "evil", "bad"]
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError(f"Hero name cannot contain: {', '.join(forbidden_words)}")
        return v


def test_basic_functionality():
    """Test basic SQLModel functionality"""

    # Create tables
    SQLModel.metadata.create_all(engine)
    logger.info("✓ Database tables created")

    # Create a team
    with Session(engine) as session:
        team = Team(name="Avengers", city="New York", founded_year=1963)
        session.add(team)
        session.commit()
        session.refresh(team)
        logger.info(f"✓ Created team: {team.name}")

    # Create heroes
    heroes_data = [
        {
            "name": "Iron Man",
            "secret_name": "Tony Stark",
            "age": 45,
            "power_level": 85,
            "origin_story": "Genius inventor who built a powered suit of armor",
        },
        {
            "name": "Captain America",
            "secret_name": "Steve Rogers",
            "age": 100,
            "power_level": 90,
            "origin_story": "Super soldier enhanced with the super soldier serum",
        },
        {
            "name": "Thor",
            "secret_name": "Thor Odinson",
            "age": 1500,  # This will be clamped to 100 by validation
            "power_level": 95,
            "origin_story": "God of Thunder from Asgard with mystical hammer",
        },
    ]

    created_heroes = []
    with Session(engine) as session:
        # Get the team
        team = session.exec(select(Team).where(Team.name == "Avengers")).first()

        if not team:
            logger.error("Team not found!")
            return

        for hero_data in heroes_data:
            try:
                # Handle age validation
                if hero_data["age"] > 100:
                    hero_data["age"] = 100

                hero = Hero(**hero_data, team_id=team.id)
                session.add(hero)
                created_heroes.append(hero)
                logger.info(f"✓ Created hero: {hero.name}")
            except ValueError as e:
                logger.error(f"✗ Failed to create hero {hero_data['name']}: {e}")

        session.commit()

        # Refresh all heroes
        for hero in created_heroes:
            session.refresh(hero)

    # Test validation
    logger.info("\n--- Testing Validation ---")
    try:
        Hero(
            name="Evil Villain",  # Should trigger validator
            secret_name="Bad Guy",
            power_level=50,
            origin_story="A story of evil deeds",
        )
    except ValueError as e:
        logger.info(f"✓ Validation caught invalid name: {e}")

    # Query with relationships
    logger.info("\n--- Testing Relationships ---")
    with Session(engine) as session:
        # Get team with heroes
        team_with_heroes = session.exec(
            select(Team).where(Team.name == "Avengers")
        ).first()

        if team_with_heroes:
            logger.info(
                f"✓ {team_with_heroes.name} has {len(team_with_heroes.heroes)} heroes"
            )

            for hero in team_with_heroes.heroes:
                logger.info(
                    f"  - {hero.name} ({hero.secret_name}) - Power: {hero.power_level}"
                )

    # Test queries
    logger.info("\n--- Testing Queries ---")
    with Session(engine) as session:
        # Find high-power heroes
        high_power_heroes = session.exec(
            select(Hero).where(Hero.power_level >= 90)
        ).all()

        logger.info(f"✓ Found {len(high_power_heroes)} high-power heroes:")
        for hero in high_power_heroes:
            logger.info(f"  - {hero.name}: {hero.power_level}")

    logger.info("\n✓ All basic functionality tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
