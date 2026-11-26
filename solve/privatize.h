/* Private names for the shared libraries of a cnlp store (`cnlp-solve --privatize`).
 *
 * A library bundle carries its own copies of common libraries (OpenSSL, curl, libstdc++,
 * ...). The dynamic loader matches libraries by name, so when the host process has already
 * loaded its own copy under the same name (Python's hashlib loads the system OpenSSL, say),
 * the bundle is handed that copy and may fail to load. `cnlp install` therefore renames them:
 * in every ELF shared library of the store, each name that belongs to a library the store
 * carries (DT_SONAME, DT_NEEDED, and the file names of its version needs) changes from
 * `lib...` to `cnl...`. The new name has the same length, so the string table is edited in
 * place; a link under the new name is added wherever the old name is, and the files keep
 * their names, so whatever opens them by path still finds them. Running it again changes
 * nothing. ELF64 little-endian (Linux x86_64, aarch64); other files are left alone.
 */
#include <fcntl.h>
#include <ftw.h>
#include <sys/mman.h>
#include <unistd.h>

#define PV_OLD "lib"
#define PV_NEW "cnl"

typedef struct { char** v; int n, cap; } pv_list;
static void pv_push(pv_list* l, const char* s) {
    if (l->n == l->cap) { l->cap = l->cap ? 2 * l->cap : 64; l->v = realloc(l->v, l->cap * sizeof *l->v); }
    l->v[l->n++] = strdup(s);
}
static int pv_has(const pv_list* l, const char* s) {
    for (int k = 0; k < l->n; k++) if (!strcmp(l->v[k], s)) return 1;
    return 0;
}

static pv_list PV_FILES, PV_NAMES;   /* the ELF files under the roots; the names they carry */

static int pv_is_elf(const char* p) {
    unsigned char h[6]; int fd = open(p, O_RDONLY); if (fd < 0) return 0;
    int ok = read(fd, h, 6) == 6 && !memcmp(h, "\177ELF", 4) && h[4] == 2 && h[5] == 1;   /* ELF64 LE */
    close(fd); return ok;
}
static int pv_collect(const char* p, const struct stat* st, int type, struct FTW* f) {
    (void)st; (void)f;
    if (type == FTW_F && strstr(p, ".so") && pv_is_elf(p)) pv_push(&PV_FILES, p);
    return 0;
}

/* The dynamic section of a mapped ELF64 file: the file offsets of the string table and of the
 * strings named by DT_SONAME / DT_NEEDED / version-need file names. Returns the count. */
typedef struct { uint32_t type, flags; uint64_t off, vaddr, paddr, filesz, memsz, align; } pv_phdr;
static int64_t pv_off(const unsigned char* b, uint64_t phoff, int phnum, uint64_t vaddr) {
    for (int k = 0; k < phnum; k++) {
        const pv_phdr* h = (const pv_phdr*)(b + phoff + (uint64_t)k * sizeof(pv_phdr));
        if (h->type == 1 && h->vaddr <= vaddr && vaddr < h->vaddr + h->filesz) return (int64_t)(h->off + vaddr - h->vaddr);
    }
    return -1;
}
static int pv_strings(const unsigned char* b, size_t size, uint64_t* out, int cap, uint64_t* soname) {
    uint64_t phoff; uint16_t phnum; memcpy(&phoff, b + 0x20, 8); memcpy(&phnum, b + 0x38, 2);
    if (phoff + (uint64_t)phnum * sizeof(pv_phdr) > size) return 0;
    const pv_phdr* dyn = NULL;
    for (int k = 0; k < phnum; k++) { const pv_phdr* h = (const pv_phdr*)(b + phoff + (uint64_t)k * sizeof(pv_phdr)); if (h->type == 2) dyn = h; }
    if (!dyn || dyn->off + dyn->filesz > size) return 0;
    uint64_t strtab = 0, verneed = 0, nverneed = 0, refs[256]; int nref = 0, sref = -1;
    for (uint64_t o = dyn->off; o + 16 <= dyn->off + dyn->filesz; o += 16) {
        int64_t tag; uint64_t val; memcpy(&tag, b + o, 8); memcpy(&val, b + o + 8, 8);
        if (tag == 0) break;
        if (tag == 5) strtab = val;
        else if ((tag == 1 || tag == 14) && nref < 256) { if (tag == 14) sref = nref; refs[nref++] = val; }
        else if (tag == 0x6FFFFFFE) verneed = val;
        else if (tag == 0x6FFFFFFF) nverneed = val;
    }
    if (!strtab) return 0;
    int64_t st = pv_off(b, phoff, phnum, strtab); if (st < 0) return 0;
    if (verneed) {   /* Elf64_Verneed: vn_version, vn_cnt (16 bits), vn_file, vn_aux, vn_next (32 bits) */
        int64_t o = pv_off(b, phoff, phnum, verneed);
        for (uint64_t k = 0; o >= 0 && k < nverneed && nref < 256 && (uint64_t)o + 16 <= size; k++) {
            uint32_t file, next; memcpy(&file, b + o + 4, 4); memcpy(&next, b + o + 12, 4);
            refs[nref++] = file;
            if (!next) break;
            o += next;
        }
    }
    int n = 0;
    for (int k = 0; k < nref && n < cap; k++) {
        uint64_t s = (uint64_t)st + refs[k];
        if (s < size) { if (k == sref && soname) *soname = s; out[n++] = s; }
    }
    return n;
}

/* map a file; NULL when it cannot be read */
static unsigned char* pv_map(const char* p, size_t* size) {
    int fd = open(p, O_RDONLY); if (fd < 0) return NULL;
    struct stat st; if (fstat(fd, &st) || st.st_size < 64) { close(fd); return NULL; }
    void* m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) return NULL;
    *size = st.st_size; return m;
}

/* the name a library is known by, as it was before any renaming */
static void pv_names(const char* p) {
    size_t size; unsigned char* b = pv_map(p, &size); if (!b) return;
    uint64_t offs[256], so = 0; pv_strings(b, size, offs, 256, &so);
    if (so && so < size) {
        const char* s = (const char*)b + so;
        if (strnlen(s, size - so) < size - so && (!strncmp(s, PV_OLD, 3) || !strncmp(s, PV_NEW, 3))) {
            char name[512]; snprintf(name, sizeof name, "%s%s", PV_OLD, s + 3);
            if (!pv_has(&PV_NAMES, name)) pv_push(&PV_NAMES, name);
        }
    }
    munmap(b, size);
}

/* rename, in one file, every name the store carries; 1 if the file changed */
static int pv_edit(const char* p) {
    size_t size; unsigned char* b = pv_map(p, &size); if (!b) return 0;
    uint64_t offs[256]; int n = pv_strings(b, size, offs, 256, NULL), edits = 0;
    uint64_t at[256];
    for (int k = 0; k < n; k++) {
        const char* s = (const char*)b + offs[k];
        if (strnlen(s, size - offs[k]) < size - offs[k] && pv_has(&PV_NAMES, s)) at[edits++] = offs[k];
    }
    if (!edits) { munmap(b, size); return 0; }
    /* written to a new file and renamed over the old: a process that has it open keeps its copy */
    unsigned char* c = malloc(size);
    if (!c) { munmap(b, size); die("out of memory"); }
    memcpy(c, b, size); munmap(b, size);
    for (int k = 0; k < edits; k++) memcpy(c + at[k], PV_NEW, 3);
    char tmp[PATH_MAX]; snprintf(tmp, sizeof tmp, "%s.cnlp-priv", p);
    struct stat st; stat(p, &st);
    FILE* f = fopen(tmp, "wb");
    if (!f) { free(c); die("cannot write %s", tmp); }
    size_t w = fwrite(c, 1, size, f); free(c);
    if (fclose(f) || w != size) die("cannot write %s", tmp);
    chmod(tmp, st.st_mode & 07777);
    if (rename(tmp, p)) die("cannot replace %s", p);
    return 1;
}

/* a link under the new name beside every entry with an old one */
static int PV_LINKS;
static int pv_link(const char* p, const struct stat* st, int type, struct FTW* f) {
    (void)st; (void)type;
    const char* base = p + f->base;
    if (!pv_has(&PV_NAMES, base)) return 0;
    char link[PATH_MAX]; snprintf(link, sizeof link, "%.*s%s%s", f->base, p, PV_NEW, base + 3);
    struct stat ls;
    if (lstat(link, &ls) && !symlink(base, link)) PV_LINKS++;
    return 0;
}

/* `cnlp-solve --privatize [DIR...]`: the store's shared runtime ($CNLPLIB_HOME/artifacts) and
 * the given library or solver directories; with none, every library and solver installed */
static void pv_installed(pv_list* dirs, const char* kind) {
    char d[PATH_MAX]; snprintf(d, sizeof d, "%s/%s", home_dir(), kind);
    DIR* dd = opendir(d); if (!dd) return;
    for (struct dirent* e; (e = readdir(dd));) {
        if (e->d_name[0] == '.') continue;
        char p[PATH_MAX]; snprintf(p, sizeof p, "%s/%s/lib", d, e->d_name);
        if (isdir(p)) pv_push(dirs, p);
    }
    closedir(dd);
}
static int privatize(int ndirs, char** given) {
    pv_list dirs = { 0 };
    char art[PATH_MAX]; snprintf(art, sizeof art, "%s/artifacts", home_dir());
    pv_push(&dirs, art);
    if (ndirs) for (int k = 0; k < ndirs; k++) pv_push(&dirs, given[k]);
    else { pv_installed(&dirs, "lib"); pv_installed(&dirs, "solvers"); }
    for (int k = 0; k < dirs.n; k++) if (isdir(dirs.v[k])) nftw(dirs.v[k], pv_collect, 32, FTW_PHYS);
    for (int k = 0; k < PV_FILES.n; k++) pv_names(PV_FILES.v[k]);
    int changed = 0;
    for (int k = 0; k < PV_FILES.n; k++) changed += pv_edit(PV_FILES.v[k]);
    for (int k = 0; k < dirs.n; k++) if (isdir(dirs.v[k])) nftw(dirs.v[k], pv_link, 32, FTW_PHYS);
    if (changed || PV_LINKS) printf("Gave %d bundled libraries private names (%d files, %d links)\n", PV_NAMES.n, changed, PV_LINKS);
    return 0;
}
