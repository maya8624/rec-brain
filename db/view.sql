-- =============================================================================
-- v_listings
-- Pre-joins listings, properties, property_addresses, property_types,
-- agents, and agencies into a single flat view for property search queries.
--
-- Usage:
--   SELECT * FROM v_listings WHERE suburb ILIKE '%Shepparton%'
--   SELECT * FROM v_listings WHERE bedrooms = 3 AND price < 800000
--   SELECT * FROM v_listings WHERE property_type = 'House' AND state = 'VIC'
--
-- =============================================================================

CREATE OR REPLACE VIEW v_listings AS
SELECT
    -- Listing
    l.id                        AS listing_id,
    l.listing_type,
    l.status                    AS listing_status,
    l.price,
    l.is_published,
    l.listed_at_utc,
    l.available_from_utc,

    -- Property
    p.id                        AS property_id,
    p.title,
    p.description,
    p.bedrooms,
    p.bathrooms,
    p.car_spaces,
    p.land_size_sqm,
    p.building_size_sqm,
    p.year_built,
    p.is_active,

    -- Property Type
    pt.name                     AS property_type,

    -- Address
    pa.address_line1,
    pa.address_line2,
    pa.suburb,
    pa.state,
    pa.postcode,
    pa.country,
    pa.latitude,
    pa.longitude,

    -- Agent
    ag.id                       AS agent_id,
    ag.first_name               AS agent_first_name,
    ag.last_name                AS agent_last_name,
    ag.email                    AS agent_email,
    ag.phone_number             AS agent_phone,

    -- Agency
    agc.id                      AS agency_id,
    agc.name                    AS agency_name,
    agc.email                   AS agency_email,
    agc.phone_number            AS agency_phone

FROM listings l
JOIN properties p           ON l.property_id    = p.id
JOIN property_addresses pa  ON pa.property_id   = p.id
JOIN property_types pt      ON pt.id            = p.property_type_id
LEFT JOIN agents ag         ON l.agent_id       = ag.id
LEFT JOIN agencies agc      ON l.agency_id      = agc.id;