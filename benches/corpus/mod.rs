//! Deterministic, production-shaped corpus for `bitcut` benchmarks.
//!
//! A profile is a CBOR document: a counterparty dictionary followed by a log of
//! money operations. Mutations mirror the production update shapes — inserts at
//! arbitrary dictionary positions, length-changing status edits scattered
//! through the log, and appends at the end.
//!
//! Generation is seeded and allocation-order independent, so the same `Params`
//! always yields byte-identical documents across machines and runs.

#![allow(dead_code)]

/// Corpus dimensions. Defaults match the production p95 shape described in the
/// task: 500 counterparties, 12k operations, ~1.5 MB uncompressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub counterparties: usize,
    pub operations: usize,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            counterparties: 500,
            operations: 12_000,
            seed: 0x5EED_1234_5EED_1234,
        }
    }
}

/// A production update shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// Three counterparties inserted at arbitrary dictionary positions.
    DictionaryInsert,
    /// Sixty length-changing status edits scattered through the log.
    StatusEdits,
    /// Forty operations appended at the end.
    Append,
    /// All three at once.
    Combined,
}

impl Scenario {
    pub const ALL: [Scenario; 4] = [
        Scenario::DictionaryInsert,
        Scenario::StatusEdits,
        Scenario::Append,
        Scenario::Combined,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Scenario::DictionaryInsert => "dictionary_insert",
            Scenario::StatusEdits => "status_edits",
            Scenario::Append => "append",
            Scenario::Combined => "combined",
        }
    }

    /// Produce the updated document for this scenario.
    pub fn apply(self, params: Params) -> Vec<u8> {
        let mut profile = Profile::generate(params);
        match self {
            Scenario::DictionaryInsert => profile.insert_counterparties(params, 3),
            Scenario::StatusEdits => profile.edit_statuses(params, 60),
            Scenario::Append => profile.append_operations(params, 40),
            Scenario::Combined => {
                profile.insert_counterparties(params, 3);
                profile.edit_statuses(params, 60);
                profile.append_operations(params, 40);
            }
        }
        profile.encode()
    }
}

/// The unmodified base document.
pub fn base_profile(params: Params) -> Vec<u8> {
    Profile::generate(params).encode()
}

/// `(old, new)` pair for a scenario.
pub fn pair(scenario: Scenario, params: Params) -> (Vec<u8>, Vec<u8>) {
    (base_profile(params), scenario.apply(params))
}

// ---------------------------------------------------------------------------
// Document model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Counterparty {
    id: u64,
    name: String,
    inn: String,
    account: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Done,
    Pending,
    Reversed,
}

impl Status {
    fn text(self) -> &'static str {
        match self {
            Status::Done => "done",
            Status::Pending => "pending",
            Status::Reversed => "reversed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Operation {
    timestamp: String,
    amount: u64,
    counterparty: u64,
    purpose: String,
    status: Status,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Profile {
    counterparties: Vec<Counterparty>,
    operations: Vec<Operation>,
}

impl Profile {
    fn generate(params: Params) -> Self {
        let mut rng = Rng::new(params.seed);
        let counterparties = (0..params.counterparties)
            .map(|i| make_counterparty(&mut rng, i))
            .collect::<Vec<_>>();
        let operations = (0..params.operations)
            .map(|i| make_operation(&mut rng, i, params.counterparties))
            .collect::<Vec<_>>();
        Self {
            counterparties,
            operations,
        }
    }

    /// Insert new counterparties at arbitrary positions. Everything after each
    /// insertion point shifts; the dictionary array header also changes width
    /// when it crosses a CBOR size class.
    fn insert_counterparties(&mut self, params: Params, count: usize) {
        let mut rng = Rng::new(params.seed ^ 0xA1);
        for k in 0..count {
            let at = rng.below(self.counterparties.len().saturating_add(1));
            let cp = make_counterparty(&mut rng, params.counterparties.saturating_add(k));
            self.counterparties.insert(at, cp);
        }
    }

    /// Flip scattered operations to a *longer* status, so the drift is
    /// piecewise constant with non-zero steps.
    fn edit_statuses(&mut self, params: Params, count: usize) {
        if self.operations.is_empty() {
            return;
        }
        let mut rng = Rng::new(params.seed ^ 0xB2);
        for _ in 0..count {
            let at = rng.below(self.operations.len());
            if let Some(op) = self.operations.get_mut(at) {
                op.status = Status::Reversed;
            }
        }
    }

    fn append_operations(&mut self, params: Params, count: usize) {
        let mut rng = Rng::new(params.seed ^ 0xC3);
        for k in 0..count {
            let index = params.operations.saturating_add(k);
            let op = make_operation(&mut rng, index, self.counterparties.len());
            self.operations.push(op);
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 << 21);
        cbor::map(&mut out, 2);

        cbor::text(&mut out, "counterparties");
        cbor::array(&mut out, as_u64(self.counterparties.len()));
        for cp in &self.counterparties {
            cbor::map(&mut out, 4);
            cbor::text(&mut out, "id");
            cbor::uint(&mut out, cp.id);
            cbor::text(&mut out, "name");
            cbor::text(&mut out, &cp.name);
            cbor::text(&mut out, "inn");
            cbor::text(&mut out, &cp.inn);
            cbor::text(&mut out, "acc");
            cbor::text(&mut out, &cp.account);
        }

        cbor::text(&mut out, "operations");
        cbor::array(&mut out, as_u64(self.operations.len()));
        for op in &self.operations {
            cbor::map(&mut out, 5);
            cbor::text(&mut out, "ts");
            cbor::text(&mut out, &op.timestamp);
            cbor::text(&mut out, "amt");
            cbor::uint(&mut out, op.amount);
            cbor::text(&mut out, "cp");
            cbor::uint(&mut out, op.counterparty);
            cbor::text(&mut out, "purpose");
            cbor::text(&mut out, &op.purpose);
            cbor::text(&mut out, "status");
            cbor::text(&mut out, op.status.text());
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Field synthesis
// ---------------------------------------------------------------------------

const LEGAL_FORMS: [&str; 4] = ["OOO", "AO", "PAO", "IP"];
const STEMS: [&str; 16] = [
    "Stroy", "Torg", "Prom", "Trans", "Agro", "Neft", "Tekh", "Info", "Med", "Energo", "Metall",
    "Logist", "Finans", "Servis", "Konsalt", "Grupp",
];
const SUFFIXES: [&str; 8] = [
    "invest", "holding", "market", "servis", "master", "trade", "profi", "resurs",
];
const PURPOSE_TEMPLATES: [&str; 8] = [
    "Oplata po schetu N{n} ot {d} za uslugi. NDS 20% - {v} rub.",
    "Perevod sobstvennyh sredstv po dogovoru N{n} ot {d}. Bez NDS.",
    "Oplata po dogovoru postavki N{n} ot {d}. V t.ch. NDS 20% - {v} rub.",
    "Vozvrat oshibochno perechislennyh sredstv po p/p N{n} ot {d}. Bez NDS.",
    "Zarabotnaya plata za period {d}, reestr N{n}. NDS ne oblagaetsya.",
    "Oplata arendy pomescheniya po dogovoru N{n} ot {d}. NDS 20% - {v} rub.",
    "Komissiya banka za obsluzhivanie scheta za {d}. Bez NDS.",
    "Oplata transportnyh uslug po zayavke N{n} ot {d}. NDS 20% - {v} rub.",
];

fn pick<'a>(rng: &mut Rng, table: &'a [&'a str]) -> &'a str {
    table.get(rng.below(table.len())).copied().unwrap_or("")
}

fn make_counterparty(rng: &mut Rng, index: usize) -> Counterparty {
    let form = pick(rng, &LEGAL_FORMS);
    let stem = pick(rng, &STEMS);
    let suffix = pick(rng, &SUFFIXES);
    Counterparty {
        id: 100_000_u64.wrapping_add(as_u64(index)),
        name: format!("{form} \"{stem}{suffix}\""),
        inn: format!(
            "{:010}",
            7_700_000_000_u64.wrapping_add(rng.next_u64() % 99_999_999)
        ),
        account: format!("40702810{:012}", rng.next_u64() % 1_000_000_000_000_u64),
    }
}

fn make_operation(rng: &mut Rng, index: usize, counterparties: usize) -> Operation {
    let n = as_u64(index);
    let day = 1_u64.wrapping_add(n % 28);
    let month = 1_u64.wrapping_add((n / 28) % 12);
    let year = 2023_u64.wrapping_add(n / 336);
    let hour = rng.next_u64() % 24;
    let minute = rng.next_u64() % 60;
    let second = rng.next_u64() % 60;

    let amount = 1_000_u64.wrapping_add(rng.next_u64() % 5_000_000);
    let doc_number = 1_u64.wrapping_add(rng.next_u64() % 99_999);
    let vat = amount / 6;

    let template = pick(rng, &PURPOSE_TEMPLATES);
    let purpose = template
        .replace("{n}", &doc_number.to_string())
        .replace("{d}", &format!("{day:02}.{month:02}.{year}"))
        .replace("{v}", &format!("{}.{:02}", vat / 100, vat % 100));

    let status = match rng.next_u64() % 10 {
        0 => Status::Pending,
        1 => Status::Reversed,
        _ => Status::Done,
    };

    Operation {
        timestamp: format!("{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"),
        amount,
        counterparty: 100_000_u64.wrapping_add(as_u64(rng.below(counterparties.max(1)))),
        purpose,
        status,
    }
}

pub fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// xorshift64* — deterministic, no dependency
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 0x1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x.wrapping_shr(12);
        x ^= x.wrapping_shl(25);
        x ^= x.wrapping_shr(27);
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        let bound = as_u64(n);
        let drawn = self.next_u64().checked_rem(bound).unwrap_or(0);
        usize::try_from(drawn).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Minimal CBOR writer (definite lengths only)
// ---------------------------------------------------------------------------

mod cbor {
    /// Major type in its encoded position (bits 5-7), so no shift is needed.
    const UINT: u8 = 0x00;
    const TEXT: u8 = 0x60;
    const ARRAY: u8 = 0x80;
    const MAP: u8 = 0xA0;

    /// Additional-information values that select a trailing length field.
    const ONE_BYTE: u8 = 0x18;
    const TWO_BYTES: u8 = 0x19;
    const FOUR_BYTES: u8 = 0x1A;
    const EIGHT_BYTES: u8 = 0x1B;

    fn head(out: &mut Vec<u8>, major: u8, value: u64) {
        if let Ok(byte) = u8::try_from(value) {
            if byte < ONE_BYTE {
                out.push(major | byte);
            } else {
                out.push(major | ONE_BYTE);
                out.push(byte);
            }
        } else if let Ok(short) = u16::try_from(value) {
            out.push(major | TWO_BYTES);
            out.extend_from_slice(&short.to_be_bytes());
        } else if let Ok(word) = u32::try_from(value) {
            out.push(major | FOUR_BYTES);
            out.extend_from_slice(&word.to_be_bytes());
        } else {
            out.push(major | EIGHT_BYTES);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }

    pub fn uint(out: &mut Vec<u8>, value: u64) {
        head(out, UINT, value);
    }

    pub fn text(out: &mut Vec<u8>, value: &str) {
        head(out, TEXT, super::as_u64(value.len()));
        out.extend_from_slice(value.as_bytes());
    }

    pub fn array(out: &mut Vec<u8>, len: u64) {
        head(out, ARRAY, len);
    }

    pub fn map(out: &mut Vec<u8>, len: u64) {
        head(out, MAP, len);
    }
}
